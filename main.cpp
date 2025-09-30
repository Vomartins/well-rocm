#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>

#include <dune/common/timer.hh>

#include <suitesparse/umfpack.h>

#include <rocsparse/rocsparse.h>

#define HIP_CALL(call)                                     \
  do {                                                     \
    hipError_t err = call;                                 \
    if (hipSuccess != err) {                               \
      printf("HIP ERROR (code = %d, %s) at %s:%d\n", err,  \
             hipGetErrorString(err), __FILE__, __LINE__);  \
      exit(1);                                             \
    }                                                      \
  } while (0)

#define ROCSOLVER_CALL(call)                                                   \
  do {                                                                         \
    rocblas_status err = call;                                                 \
    if (rocblas_status_success != err) {                                       \
      printf("rocSOLVER ERROR (code = %d) at %s:%d\n", err, __FILE__,          \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define ROCSPARSE_CALL(call)                                                   \
do {                                                                           \
    rocsparse_status err = call;                                               \
    if (rocsparse_status_success != err) {                                     \
    printf("rocSPARSE ERROR (code = %d) at %s:%d\n", err, __FILE__,            \
            __LINE__);                                                         \
    exit(1);                                                                   \
    }                                                                          \
} while (0)

void checkHIPAlloc(void* ptr) {
    if (ptr == nullptr) {
        std::cerr << "HIP malloc failed." << std::endl;
        exit(1);
    }
}

template <class Scalar>
void readVector(std::vector<Scalar>& vec, std::string filename){
    std::ifstream input_file(filename);
    if (input_file.is_open()) {
        Scalar num;
        while (input_file >> num) {
            vec.push_back(num);
        }
        input_file.close();
    } else {
        std::cerr << "Error opening file." << std::endl;
    }
}

template <class Scalar>
void saveVector(std::vector<Scalar> vec, std::string filename){
    std::ofstream output_file(filename);
    if (output_file.is_open()) {
        for (const auto& num : vec) {
            output_file << num << " ";
        }
        output_file.close();
    } else {
        std::cerr << "Error opening file." << std::endl;
    }
}

template <typename T>
void printVector(const std::vector<T>& vec, const std::string& name){
  std::cout << name <<": " << size(vec) << std::endl;
  std::cout << "[ ";
  for (const auto& val : vec) std::cout << val << " ";
  std::cout << "]";
  std::cout << std::endl;
  std::cout << std::endl;
}

// Structure to hold a single non-zero element in COO format
struct Triplet {
    int row;
    int col;
    double val;
};

// Comparison function to sort triplets by row, then column
bool compareTriplets(const Triplet& a, const Triplet& b) {
    if (a.row != b.row) {
        return a.row < b.row;
    }
    return a.col < b.col;
}

bool compareTuple(const std::tuple<double, int, int>& a, const std::tuple<double, int, int>& b) {
    if (std::get<1>(a) != std::get<1>(b)) {
        return std::get<1>(a) < std::get<1>(b);
    }
    return std::get<2>(a) < std::get<2>(b);
}

void BCSRrecttoCSR(
    std::vector<double>& Bval,
    std::vector<int>& Bcol_ind,
    std::vector<int>& Brow_ptr,
    int Br, int Bc,
    std::vector<double>& val,
    std::vector<int>& col_ind,
    std::vector<int>& row_ptr) {

    //std::cout << " ------ BCSRrecttoCSR ------ " << std::endl;
    int num_br = Brow_ptr.size() - 1;
    int num_bc = *std::max_element(Bcol_ind.begin(), Bcol_ind.end()) + 1;
    int M = num_br * Br;
    int N = num_bc * Bc;
    //std::cout << "num_br = " << num_br << " -- " << "num_bc = " << num_bc << " -- " << "M = " << M << " -- " << "N = " << N << std::endl;

    row_ptr.resize(M + 1);
    row_ptr[0] = 0;

    int csr_idx = 0; // Current index for CSR arrays val and col_ind

    for (int I = 0; I < num_br; I++) {
        int block_start = Brow_ptr[I];
        int block_end = Brow_ptr[I + 1];

        for (int r = 0; r < Br; r++) {
            int i = I * Br + r;
            if ( i >= M) break;

            for (int block_idx = block_start; block_idx < block_end; block_idx++) {
                int J  = Bcol_ind[block_idx]; // Block column index

                for ( int c = 0; c < Bc; c++) {
                    int j = J * Bc + c; // Global column index
                    if (j >= N) continue;

                    //Assuming row-major block storage
                    int block_element_idx = block_idx * Br * Bc + r * Bc + c;

                    double value = Bval[block_element_idx];

                    if ( value != 0) {
                        val.push_back(value);
                        col_ind.push_back(j);
                        csr_idx++;
                    }
                }
            }
            if (i + 1 <= M){
                row_ptr[i + 1] = csr_idx;
            }
        }
    }
}

void CustomtoCSR(std::vector<double>& Dvals, std::vector<int>& Drows, std::vector<int>& Dcols, std::vector<double>& csrDvals, std::vector<int>& csrDcols, std::vector<int>& csrDrows){
    int D_m = Dcols.size() - 1;
    std::vector<int> Cols(Dvals.size());

    for (int i=0; i< Dcols.size()-1; i++){
        for (int j=Dcols[i]; j<Dcols[i+1]; j++){
            Cols[j] = i;
        }
    }

    std::vector<std::tuple<double, int, int>> COO;

    for (int i=0; i<Dvals.size(); i++){
        COO.push_back(std::make_tuple(Dvals[i], Drows[i], Cols[i]));
    }

    std::sort(COO.begin(), COO.end(), compareTuple);

    auto new_end = std::remove_if(COO.begin(), COO.end(),
        [](const auto& t) {
            // Check if the first element (index 0) is equal to zero
            return std::get<0>(t) == 0;
        });

    COO.erase(new_end, COO.end());

    csrDvals.resize(COO.size());
    csrDcols.resize(COO.size());
    csrDrows.assign(D_m + 1, 0);

    for (int i=0; i<COO.size(); i++){
        csrDvals[i] = std::get<0>(COO[i]);
        csrDcols[i] = std::get<2>(COO[i]);

        csrDrows[std::get<1>(COO[i]) + 1]++;
    }

    std::partial_sum(csrDrows.begin(), csrDrows.end(), csrDrows.begin());
}

void squareCSCtoCSR(std::vector<double> vals, std::vector<int> rows, std::vector<int> cols, std::vector<double>& vals_, std::vector<int>& rows_, std::vector<int>& cols_)
{
    int sizeDvals = size(vals);
    int sizeDcols = size(cols);

    std::vector<int> Cols(sizeDvals);

    for(int i=0; i<sizeDcols-1; i++){
      for(int j=cols[i];j<cols[i+1];j++){
        Cols[j] = i;
      }
    }

    std::vector<std::tuple<int,double,int>> ConvertVec;

    for(int i=0; i<sizeDvals; i++){
      ConvertVec.push_back(std::make_tuple(rows[i],vals[i],Cols[i]));
    }

    std::sort(ConvertVec.begin(),ConvertVec.end());

    auto it = ConvertVec.begin();
    while (it != ConvertVec.end()) {
        auto range_end = std::find_if(it, ConvertVec.end(),
            [it](const std::tuple<int, double, int>& tup) {
                return std::get<0>(tup) != std::get<0>(*it);
            });

        std::sort(it, range_end,
            [](const std::tuple<int, double, int>& a, const std::tuple<int, double, int>& b) {
                return std::get<2>(a) < std::get<2>(b);
            });

        it = range_end;
    }

    for(int i=0; i<sizeDvals; i++){
        //std::cout << "{" << std::get<0>(ConvertVec[i]) << ", "  << std::get<1>(ConvertVec[i]) << ", " << std::get<2>(ConvertVec[i]) << "}" << std::endl;
        vals_[i] = std::get<1>(ConvertVec[i]);
        cols_[i] = std::get<2>(ConvertVec[i]);
    }

    for(int target = 0; target< sizeDcols-1; target++){
      it = std::find_if(ConvertVec.begin(), ConvertVec.end(),
          [target](const std::tuple<int, double, int>& tup) {
              return std::get<0>(tup) == target;
          });

      if (it != ConvertVec.end()) {
          rows_[target] = std::distance(ConvertVec.begin(),it);
      } else {
          std::cout << target << " not found in the third element of any tuple.\n";
      }
    }

    for (int i=1; i<sizeDcols-1; i++){
        if(rows_[i] == 0){
            rows_[i] = rows_[i+1];
        }
    }
    rows_[sizeDcols-1] = cols[sizeDcols-1];
}

void CPUBx(const std::vector<double>& Bvals,
            const std::vector<int>& Bcols,
            const std::vector<int>& Brows,
            const std::vector<double>& x,
            std::vector<double>& z,
            int dim, int dim_wells, int Mb) {
    for (unsigned int row = 0; row < Mb; ++row) {
        // for every block in the row
        for (unsigned int blockID = Brows[row]; blockID < Brows[row + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim_wells; ++j) {
                double temp = 0.0;
                for (unsigned int k = 0; k < dim; ++k) {
                    temp += Bvals[blockID * dim * dim_wells + j * dim + k] * x[colIdx * dim + k];
                }
                z[row * dim_wells + j] += temp;
            }
        }
    }
}

void CPUCz(const std::vector<double>& Cvals,
            const std::vector<int>& Bcols,
            const std::vector<int>& Brows,
            const std::vector<double>& z,
            std::vector<double>& y,
            int dim, int dim_wells, int Mb) {
    for (unsigned int row = 0; row < Mb; ++row) {
        // for every block in the row
        for (unsigned int blockID = Brows[row]; blockID < Brows[row + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim; ++j) {
                double temp = 0.0;
                for (unsigned int k = 0; k < dim_wells; ++k) {
                    temp += Cvals[blockID * dim * dim_wells + j + k * dim] * z[row * dim_wells + k];
                }
                y[colIdx * dim + j] -= temp;
            }
        }
    }
}

void rocsparseBx(double* vals,
                int* cols,
                int* rows,
                double* x,
                double* y,
                int B_M,
                int B_N,
                int B_nnz,
                rocsparse_handle handle,
                rocsparse_mat_descr descr_B,
                rocsparse_mat_info B_info) {
    double alpha = 1.0;
    double beta = 0.0;

    ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(handle, rocsparse_operation_none, B_M, B_N, B_nnz, descr_B, vals, rows, cols, B_info));

    ROCSPARSE_CALL(rocsparse_dcsrmv(handle, rocsparse_operation_none, B_M, B_N, B_nnz, &alpha, descr_B, vals, rows, cols, B_info, x, &beta, y));

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}

// Method for operation y = y - C_w^T * x with rocsparse method, C_w must be in CSR format
void rocsparseCz(double* vals,
                int* cols,
                int* rows,
                double* x,
                double* y,
                int C_M,
                int C_N,
                int C_nnz,
                rocsparse_handle handle,
                rocsparse_mat_descr descr_C,
                rocsparse_mat_info C_info) {
    double alpha = -1.0;
    double beta = 1.0;

    ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(handle, rocsparse_operation_transpose, C_M, C_N, C_nnz, descr_C, vals, rows, cols, C_info));

    ROCSPARSE_CALL(rocsparse_dcsrmv(handle, rocsparse_operation_transpose, C_M, C_N, C_nnz, &alpha, descr_C, vals, rows, cols, C_info, x, &beta, y));

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
}


int main(int argc, char ** argv)
{
    void *UMFPACK_Symbolic, *UMFPACK_Numeric;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> Dvals;
    std::vector<int> Dcols;
    std::vector<int> Drows;
    std::vector<double> Bvals;
    std::vector<int> Bcols;
    std::vector<int> Brows;
    std::vector<double> Cvals;

    readVector(Dvals, "Dvals.txt");
    readVector(Dcols, "Dcols.txt");
    readVector(Drows, "Drows.txt");
    readVector(Bvals, "Bvals.txt");
    readVector(Bcols, "Bcols.txt");
    readVector(Brows, "Brows.txt");
    readVector(Cvals, "Cvals.txt");
    readVector(x, "vecx.txt");
    readVector(y, "vecy.txt");

    std::cout << "size(x) = " << x.size() << std::endl;
    std::cout << "size(y) = " << y.size() << std::endl;

    int block_m = 4;
    int block_n = 3;

    int B_m = Brows.size() - 1;
    int B_n = *std::max_element(Bcols.begin(), Bcols.end()) + 1;
    std::cout << "B_m = " << B_m << std::endl;

    int M = B_m * block_m;
    std::cout << "M = " << M << std::endl;

    umfpack_di_symbolic(M, M, Dcols.data(), Drows.data(), Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
    umfpack_di_numeric(Dcols.data(), Drows.data(), Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);

    std::vector<double> z1(B_m * block_m);
    std::vector<double> z2(B_m * block_m);

    std::fill(z1.begin(), z1.end(), 0.0);
    std::fill(z2.begin(), z2.end(), 0.0);

    CPUBx(Bvals, Bcols, Brows, x, z1, block_n, block_m, B_m);

    printVector(z1, "z1 CPU");

    umfpack_di_solve(UMFPACK_A, Dcols.data(), Drows.data(), Dvals.data(), z2.data(), z1.data(), UMFPACK_Numeric, nullptr, nullptr);

    printVector(z2, "z2 CPU");

    CPUCz(Cvals, Bcols, Brows, z2, y, block_n, block_m, B_m);

    printVector(y, "y CPU");

    // std::cout << B_m << " x " << B_n << std::endl;

    // std::cout << Bvals.size() << std::endl;
    // std::cout << Bcols.size() << std::endl;
    // std::cout << Brows.size() << std::endl;

    // printVector(Cvals, "Cvals");
    // printVector(Bvals, "Bvals");
    // printVector(Bcols, "Bcols");
    // printVector(Brows, "Brows");

    // std::cout << "##################################################################" << std::endl;

    std::vector<double> csrBvals;
    std::vector<int> csrBcols;
    std::vector<int> csrBrows;

    std::vector<double> csrCvals;
    std::vector<int> csrCcols;
    std::vector<int> csrCrows;

    std::vector<double> csrDvals;
    std::vector<int> csrDcols;
    std::vector<int> csrDrows;

    //squareCSCtoCSR(Dvals, Drows, Dcols, csrDvals, csrDrows, csrDcols); // Segmentation fault (?)
    CustomtoCSR(Dvals, Drows, Dcols, csrDvals, csrDcols, csrDrows);

    BCSRrecttoCSR(Bvals, Bcols, Brows, block_m, block_n, csrBvals, csrBcols, csrBrows);

    BCSRrecttoCSR(Cvals, Bcols, Brows, block_m, block_n, csrCvals, csrCcols, csrCrows);

    // printVector(csrBvals, "csrBvals");
    // printVector(csrBcols, "csrBcols");
    // printVector(csrBrows, "csrBrows");
    // printVector(csrCvals, "csrCvals");
    // printVector(csrCcols, "csrCcols");
    // printVector(csrCrows, "csrCrows");
    printVector(csrDvals, "csrDvals");
    printVector(csrDcols, "csrDcols");
    printVector(csrDrows, "csrDrows");

    // RocSPARSE

    double one  = 1.0;
    rocsparse_int rocM;
    rocsparse_int rocN;
    rocsparse_int Nrhs = 1;
    rocsparse_int lda;
    rocsparse_int ldb;
    rocsparse_mat_info ilu_info, B_info, C_info;
    rocsparse_mat_descr descr_M, descr_L, descr_U, descr_B, descr_C;
    std::size_t d_bufferSize_M, d_bufferSize_L, d_bufferSize_U, d_bufferSize;
    void *d_buffer;
    rocsparse_handle handle;
    rocsparse_operation operation = rocsparse_operation_none;
    rocsparse_int nnzs;
    rocsparse_int zero_position;
    rocsparse_status status;

    // Device arrays
    double *d_Dvals;
    int *d_Dcols;
    int *d_Drows;
    double *d_Bvals;
    int *d_Bcols;
    int *d_Brows;
    double *d_Cvals;
    int *d_Ccols;
    int *d_Crows;
    double *d_z;
    double *d_z_aux;
    double *d_x;
    double *d_y;

    rocM = size(csrDrows) - 1;
    ldb = rocM;
    nnzs = size(csrDvals);

    std::cout << "rocM = " << rocM << std::endl;
    std::cout << "ldb = " << ldb << std::endl;
    std::cout << "nnzs = " << nnzs << std::endl;

    HIP_CALL(hipMalloc(&d_Dvals, sizeof(double)*size(csrDvals)));
    checkHIPAlloc(d_Dvals);
    HIP_CALL(hipMalloc(&d_Dcols, sizeof(int)*size(csrDcols)));
    checkHIPAlloc(d_Dcols);
    HIP_CALL(hipMalloc(&d_Drows, sizeof(int)*size(csrDrows)));
    checkHIPAlloc(d_Drows);
    HIP_CALL(hipMalloc(&d_Bvals, sizeof(double)*size(csrBvals)));
    checkHIPAlloc(d_Bvals);
    HIP_CALL(hipMalloc(&d_Bcols, sizeof(int)*size(csrBcols)));
    checkHIPAlloc(d_Bcols);
    HIP_CALL(hipMalloc(&d_Brows, sizeof(int)*size(csrBrows)));
    checkHIPAlloc(d_Brows);
    HIP_CALL(hipMalloc(&d_Cvals, sizeof(double)*size(csrCvals)));
    checkHIPAlloc(d_Cvals);
    HIP_CALL(hipMalloc(&d_Ccols, sizeof(int)*size(csrCcols)));
    checkHIPAlloc(d_Ccols);
    HIP_CALL(hipMalloc(&d_Crows, sizeof(int)*size(csrCrows)));
    checkHIPAlloc(d_Crows);
    // HIP_CALL(hipMalloc(&d_x_elem, sizeof(double)*dim*size(Bcols)));
    // checkHIPAlloc(d_x_elem);
    HIP_CALL(hipMalloc(&d_z, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z);
    HIP_CALL(hipMalloc(&d_z_aux, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z_aux);
    HIP_CALL(hipMalloc(&d_x, sizeof(double)*size(x)));
    checkHIPAlloc(d_x);
    HIP_CALL(hipMalloc(&d_y, sizeof(double)*size(y)));
    checkHIPAlloc(d_y);

    HIP_CALL(hipMemcpy(d_Dvals, csrDvals.data(), size(csrDvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Dcols, csrDcols.data(), size(csrDcols)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Drows, csrDrows.data(), size(csrDrows)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bvals, csrBvals.data(), size(csrBvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bcols, csrBcols.data(), size(csrBcols)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Brows, csrBrows.data(), size(csrBrows)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Cvals, csrCvals.data(), size(csrCvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Ccols, csrCcols.data(), size(csrCcols)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Crows, csrCrows.data(), size(csrCrows)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_x, x.data(), size(x)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_y, y.data(), size(y)*sizeof(double), hipMemcpyHostToDevice));

    ROCSPARSE_CALL(rocsparse_create_handle(&handle));

    // Create matrix descriptor for matrix M
    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_M));
    // Matrix descriptor and info for B_w and C_w
    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_B));
    ROCSPARSE_CALL(rocsparse_create_mat_info(&B_info));
    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_C));
    ROCSPARSE_CALL(rocsparse_create_mat_info(&C_info));

    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_L));
    ROCSPARSE_CALL(rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower));
    ROCSPARSE_CALL(rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit));

    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_U));
    ROCSPARSE_CALL(rocsparse_set_mat_fill_mode(descr_U, rocsparse_fill_mode_upper));
    ROCSPARSE_CALL(rocsparse_set_mat_diag_type(descr_U, rocsparse_diag_type_non_unit));

    // Create matrix info structure
    ROCSPARSE_CALL(rocsparse_create_mat_info(&ilu_info));
    // Obtain required buffer sizes
    ROCSPARSE_CALL(rocsparse_dcsrilu0_buffer_size(handle, rocM, nnzs,
						  descr_M, d_Dvals, d_Drows, d_Dcols, ilu_info, &d_bufferSize_M));
    ROCSPARSE_CALL(rocsparse_dcsrsv_buffer_size(handle, operation, rocM, nnzs,
						descr_L, d_Dvals, d_Drows, d_Dcols, ilu_info, &d_bufferSize_L));
    ROCSPARSE_CALL(rocsparse_dcsrsv_buffer_size(handle, operation, rocM, nnzs,
						descr_U, d_Dvals, d_Drows, d_Dcols, ilu_info, &d_bufferSize_U));
    d_bufferSize = std::max(d_bufferSize_M, std::max(d_bufferSize_L, d_bufferSize_U));
    HIP_CALL(hipMalloc(&d_buffer, d_bufferSize));

    // Perform analysis steps
    ROCSPARSE_CALL(rocsparse_dcsrilu0_analysis(handle, \
                               rocM, nnzs, descr_M, d_Dvals, d_Drows, d_Dcols, \
					        ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dcsrsv_analysis(handle, operation, \
                             rocM, nnzs, descr_L, d_Dvals, d_Drows, d_Dcols, \
					      ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dcsrsv_analysis(handle, operation, \
                             rocM, nnzs, descr_U, d_Dvals, d_Drows, d_Dcols, \
					     ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));

    // Check for zero pivot
    status = rocsparse_csrilu0_zero_pivot(handle, ilu_info, &zero_position);
    if (status != rocsparse_status_success) {
        printf("--- RocSPARSE Error --- L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
    }

    ROCSPARSE_CALL(rocsparse_dcsrilu0(handle, rocM, nnzs, descr_M,
				      d_Dvals, d_Drows, d_Dcols, ilu_info, rocsparse_solve_policy_auto, d_buffer));

    status = rocsparse_csrilu0_zero_pivot(handle, ilu_info, &zero_position);
    if (status != rocsparse_status_success) {
        printf("--- RocSPARSE Error --- L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
    }

    HIP_CALL(hipMemset(d_z, 0, ldb*Nrhs*sizeof(double)));

    int B_M = csrBrows.size() - 1;
    int B_N = *std::max_element(csrBcols.begin(), csrBcols.end()) + 1;
    int B_nnz = csrBvals.size();
    rocsparseBx(d_Bvals, d_Bcols, d_Brows, d_x, d_z, B_M, B_N, B_nnz, handle, descr_B, B_info);

    std::vector<double> h_z1;
    h_z1.resize(B_M);

    HIP_CALL(hipMemcpy(h_z1.data(), d_z, sizeof(double) * B_M, hipMemcpyDeviceToHost));

    ROCSPARSE_CALL(rocsparse_dcsrsv_solve(handle, \
                              operation, rocM, nnzs, &one, \
					  descr_L, d_Dvals, d_Drows, d_Dcols,  ilu_info, d_z, d_z_aux, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dcsrsv_solve(handle,\
                              operation, rocM, nnzs, &one, \
					  descr_U, d_Dvals, d_Drows, d_Dcols, ilu_info, d_z_aux, d_z, rocsparse_solve_policy_auto, d_buffer));

    std::vector<double> h_z2;
    h_z2.resize(B_M);

    HIP_CALL(hipMemcpy(h_z2.data(), d_z, sizeof(double) * B_M, hipMemcpyDeviceToHost));

    int C_M = csrCrows.size() - 1;
    int C_N = *std::max_element(csrCcols.begin(), csrCcols.end()) + 1;
    int C_nnz = csrCvals.size();

    rocsparseCz(d_Bvals, d_Bcols, d_Brows, d_z, d_y, C_M, C_N, C_nnz, handle, descr_C, C_info);

    std::vector<double> h_y;
    h_y.resize(y.size());

    HIP_CALL(hipMemcpy(h_y.data(), d_y, sizeof(double) * B_M, hipMemcpyDeviceToHost));

    for (int i = 0; i < B_M; i++) {
        printf("index %d error = %f \n", i, h_z1[i] - z1[i]);
    }
    printf("\n");
    for (int i = 0; i < B_M; i++) {
        printf("index %d error = %f \n", i, h_z2[i] - z2[i]);
    }
    printf("\n");
    for (int i = 0; i < y.size(); i++) {
        printf("index %d error = %f \n", i, h_y[i] - y[i]);
    }

    ROCSPARSE_CALL(rocsparse_destroy_handle(handle));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_M));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_L));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_U));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_B));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_C));
    ROCSPARSE_CALL(rocsparse_destroy_mat_info(ilu_info));
    ROCSPARSE_CALL(rocsparse_destroy_mat_info(B_info));
    ROCSPARSE_CALL(rocsparse_destroy_mat_info(C_info));

    HIP_CALL(hipFree(d_Dvals));
    HIP_CALL(hipFree(d_Dcols));
    HIP_CALL(hipFree(d_Drows));
    HIP_CALL(hipFree(d_Bvals));
    HIP_CALL(hipFree(d_Bcols));
    HIP_CALL(hipFree(d_Brows));
    HIP_CALL(hipFree(d_Cvals));
    HIP_CALL(hipFree(d_Ccols));
    HIP_CALL(hipFree(d_Crows));
    HIP_CALL(hipFree(d_z));
    HIP_CALL(hipFree(d_z_aux));
    HIP_CALL(hipFree(d_x));
    HIP_CALL(hipFree(d_y));

    umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    umfpack_di_free_numeric(&UMFPACK_Numeric);

    return 0;
}
