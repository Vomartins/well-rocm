
#include "conversionData.hpp"
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

#include "linearSystemData.hpp"
#include "umfpackSolver.hpp"
#include "conversionData.hpp"

#include <dune/common/timer.hh>

#include <suitesparse/umfpack.h>

#include <rocsparse/rocsparse.h>

#include <Eigen/Dense>
#include <limits> // For infinity()

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


/**
 * @brief Calculates the infinity norm (L_infinity) of a vector.
 * * The infinity norm is the maximum absolute value of the vector's components.
 * * @param v The input vector.
 * @return The infinity norm of the vector. Returns 0.0 if the vector is empty.
 */
double infinityNorm(const std::vector<double>& v) {
    if (v.empty()) {
        return 0.0;
    }

    double max_abs = 0.0;
    for (double val : v) {
        max_abs = std::max(max_abs, std::abs(val));
    }
    return max_abs;
}

/**
 * @brief Calculates the relative error between two vectors using the infinity norm.
 * * Relative Error = ||v1 - v2||_infinity / ||v2||_infinity
 * * @param v1 The first vector.
 * @param v2 The second (reference) vector.
 * @return The relative error. Returns std::numeric_limits<double>::quiet_NaN()
 * if vectors have different sizes, and returns 0.0 if both vectors are empty.
 * If ||v2||_infinity is zero, it handles the division by zero:
 * - If ||v1 - v2||_infinity is also zero (v1 == v2), returns 0.0.
 * - Otherwise (v1 != v2 and ||v2||_infinity == 0), returns infinity.
 */
double relativeErrorInfinityNorm(const std::vector<double>& v1, const std::vector<double>& v2) {
    // 1. Check for consistent dimensions
    if (v1.size() != v2.size()) {
        std::cerr << "Error: Vectors must have the same size." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Handle empty vectors
    if (v1.empty()) {
        return 0.0;
    }

    // 2. Calculate the difference vector (v1 - v2) and its infinity norm (Absolute Error)
    double abs_error_inf_norm = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        abs_error_inf_norm = std::max(abs_error_inf_norm, std::abs(diff));
    }
    printf("abs_error = %f \n", abs_error_inf_norm);

    // 3. Calculate the infinity norm of the reference vector (v2)
    double v2_inf_norm = infinityNorm(v2);

    // 4. Handle division by zero
    if (v2_inf_norm == 0.0) {
        // If v2 is the zero vector:
        if (abs_error_inf_norm == 0.0) {
            // v1 is also the zero vector (v1 == v2), relative error is 0
            return 0.0;
        } else {
            // v1 is non-zero, v2 is zero. Relative error is infinite.
            return std::numeric_limits<double>::infinity();
        }
    }

    // 5. Calculate and return the relative error
    return abs_error_inf_norm / v2_inf_norm;
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

/**
 * @brief Performs sparse matrix-vector multiplication (SpMV) y = D * x
 * for a matrix D stored in Compressed Sparse Row (CSR) format.
 *
 * @param Dvals The non-zero values of the matrix D.
 * @param Dcols The column indices corresponding to the values in Dvals.
 * @param Drows The row pointers (starts of rows) in Dvals/Dcols.
 * @param x The dense input vector.
 * @param y The dense output vector (result). Must be pre-sized to the number of rows in D.
 * @param num_rows The number of rows in the matrix D.
 */
void spmv_csr(
    const std::vector<double>& Dvals,
    const std::vector<int>& Dcols,
    const std::vector<int>& Drows,
    const std::vector<double>& x,
    std::vector<double>& y,
    int num_rows)
{
    // The matrix D is 'num_rows' x 'num_cols'.
    // The vector x must have 'num_cols' elements.
    // The vector y must have 'num_rows' elements.
    // In this function, we assume the sizes are correct.

    // Ensure the output vector 'y' is the correct size and initialized to zero.
    // Drows.size() is num_rows + 1, so the number of rows is Drows.size() - 1.
    // We use the passed 'num_rows' for clarity.
    if (y.size() != static_cast<size_t>(num_rows)) {
        y.resize(num_rows, 0.0);
    } else {
        // Clear previous values in y for a clean multiplication
        std::fill(y.begin(), y.end(), 0.0);
    }

    // Iterate over each row of the matrix D
    for (int i = 0; i < num_rows; ++i) {
        // Drows[i] gives the index in Dvals/Dcols where row 'i' starts.
        // Drows[i+1] gives the index where row 'i' ends (and row 'i+1' begins).

        // This is the starting index in Dvals and Dcols for the current row 'i'
        int start_idx = Drows[i];

        // This is the ending index (exclusive)
        int end_idx = Drows[i+1];

        // The result element y[i] is the dot product of the i-th row of D and vector x
        double row_sum = 0.0;

        // Iterate over the non-zero elements in the current row
        for (int j = start_idx; j < end_idx; ++j) {
            // Dvals[j] is the non-zero value D(i, Dcols[j])
            // Dcols[j] is the column index 'k' for this non-zero value
            // We compute: row_sum += D(i, k) * x[k]

            double value = Dvals[j];
            int col_index = Dcols[j];

            row_sum += value * x[col_index];
        }

        // Store the result for the current row
        y[i] = row_sum;
    }
}


int main(int argc, char ** argv)
{
    int block_m = 4;
    int block_n = 3;

    WellSolver::LinearSystemData data(block_m, block_n);

    data.printDataSizes();

    WellSolver::UMFPACKSolver umfpackSolver(data);

    umfpackSolver.cpuBx();
    // data.printVector(umfpackSolver.z1, "z1 UMFPACK");

    umfpackSolver.solveSystem();
    // data.printVector(umfpackSolver.z2, "z2 UMFPACK");

    umfpackSolver.cpuCz();
    // data.printVector(umfpackSolver.y, "y UMFPACK");

    // Data sparse storage format convertion

    WellSolver::ConversionData convData(data);

    //squareCSCtoCSR(Dvals, Drows, Dcols, csrDvals, csrDrows, csrDcols); // Segmentation fault (?)
    convData.CustomtoCSR();

    convData.ConvertB();

    convData.ConvertC();

    convData.printDataSizes();

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

    rocM = size(convData.csrDrows) - 1;
    ldb = rocM;
    nnzs = size(convData.csrDvals);

    std::cout << "rocM = " << rocM << std::endl;
    std::cout << "ldb = " << ldb << std::endl;
    std::cout << "nnzs = " << nnzs << std::endl;

    HIP_CALL(hipMalloc(&d_Dvals, sizeof(double)*size(convData.csrDvals)));
    checkHIPAlloc(d_Dvals);
    HIP_CALL(hipMalloc(&d_Dcols, sizeof(int)*size(convData.csrDcols)));
    checkHIPAlloc(d_Dcols);
    HIP_CALL(hipMalloc(&d_Drows, sizeof(int)*size(convData.csrDrows)));
    checkHIPAlloc(d_Drows);
    HIP_CALL(hipMalloc(&d_Bvals, sizeof(double)*size(convData.csrBvals)));
    checkHIPAlloc(d_Bvals);
    HIP_CALL(hipMalloc(&d_Bcols, sizeof(int)*size(convData.csrBcols)));
    checkHIPAlloc(d_Bcols);
    HIP_CALL(hipMalloc(&d_Brows, sizeof(int)*size(convData.csrBrows)));
    checkHIPAlloc(d_Brows);
    HIP_CALL(hipMalloc(&d_Cvals, sizeof(double)*size(convData.csrCvals)));
    checkHIPAlloc(d_Cvals);
    HIP_CALL(hipMalloc(&d_Ccols, sizeof(int)*size(convData.csrCcols)));
    checkHIPAlloc(d_Ccols);
    HIP_CALL(hipMalloc(&d_Crows, sizeof(int)*size(convData.csrCrows)));
    checkHIPAlloc(d_Crows);
    // HIP_CALL(hipMalloc(&d_x_elem, sizeof(double)*dim*size(Bcols)));
    // checkHIPAlloc(d_x_elem);
    HIP_CALL(hipMalloc(&d_z, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z);
    HIP_CALL(hipMalloc(&d_z_aux, sizeof(double)*ldb*Nrhs));
    checkHIPAlloc(d_z_aux);
    HIP_CALL(hipMalloc(&d_x, sizeof(double)*size(data.x)));
    checkHIPAlloc(d_x);
    HIP_CALL(hipMalloc(&d_y, sizeof(double)*size(data.y)));
    checkHIPAlloc(d_y);

    HIP_CALL(hipMemcpy(d_Dvals, convData.csrDvals.data(), size(convData.csrDvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Dcols, convData.csrDcols.data(), size(convData.csrDcols)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Drows, convData.csrDrows.data(), size(convData.csrDrows)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bvals, convData.csrBvals.data(), size(convData.csrBvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bcols, convData.csrBcols.data(), size(convData.csrBcols)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Brows, convData.csrBrows.data(), size(convData.csrBrows)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Cvals, convData.csrCvals.data(), size(convData.csrCvals)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Ccols, convData.csrCcols.data(), size(convData.csrCcols)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Crows, convData.csrCrows.data(), size(convData.csrCrows)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_x, data.x.data(), size(data.x)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_y, data.y.data(), size(data.y)*sizeof(double), hipMemcpyHostToDevice));

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

    int B_M = convData.csrBrows.size() - 1;
    int B_N = *std::max_element(convData.csrBcols.begin(), convData.csrBcols.end()) + 1;
    int B_nnz = convData.csrBvals.size();
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

    int C_M = convData.csrCrows.size() - 1;
    int C_N = *std::max_element(convData.csrCcols.begin(), convData.csrCcols.end()) + 1;
    int C_nnz = convData.csrCvals.size();

    rocsparseCz(d_Bvals, d_Bcols, d_Brows, d_z, d_y, C_M, C_N, C_nnz, handle, descr_C, C_info);

    std::vector<double> h_y;
    h_y.resize(data.y.size());

    HIP_CALL(hipMemcpy(h_y.data(), d_y, sizeof(double) * B_M, hipMemcpyDeviceToHost));

    double rel_err;
    printf("--- Relative error h_z1 and z1 --- \n");
    rel_err = relativeErrorInfinityNorm(h_z1, umfpackSolver.z1);
    printf("rel_error = %f \n", rel_err);

    printf("--- Relative error h_z2 and z2 --- \n");
    rel_err = relativeErrorInfinityNorm(h_z2, umfpackSolver.z2);
    printf("rel_error = %f \n", rel_err);

    printf("--- Relative error h_y and y --- \n");
    rel_err = relativeErrorInfinityNorm(h_y, umfpackSolver.y);
    printf("rel_error = %f \n", rel_err);

    std::vector<double> Dz;
    Dz.resize(umfpackSolver.z1.size());

    printf("--- Infinity norm of residual of D_w z = B_w x (RocSPARSE) --- \n");
    spmv_csr(convData.csrDvals, convData.csrDcols, convData.csrDrows, h_z2, Dz, convData.csrDrows.size() - 1);
    rel_err = relativeErrorInfinityNorm(Dz, h_z1);
    printf("|Dz-Bx| = %f \n", rel_err);

    printf("--- Infinity norm of residual of D_w z = B_w x (UMFPACK) --- \n");
    spmv_csr(convData.csrDvals, convData.csrDcols, convData.csrDrows, umfpackSolver.z2, Dz, convData.csrDrows.size() - 1);
    rel_err = relativeErrorInfinityNorm(Dz, umfpackSolver.z1);
    printf("|Dz-Bx| = %f \n", rel_err);

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

    return 0;
}
