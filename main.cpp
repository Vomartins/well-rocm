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

void removeZerosFromCSCInPlace(
    std::vector<double>& A_values,
    std::vector<int>& A_row_ind,
    std::vector<int>& A_col_ptr) {

    // Number of columns and old number of non-zero elements
    int num_cols = A_col_ptr.size() - 1;
    int old_nnz = A_values.size();

    // The new number of non-zero elements
    int new_nnz = 0;

    // Create a temporary vector to store the new column pointers
    std::vector<int> temp_col_ptr(num_cols + 1, 0);

    for (int j = 0; j < num_cols; ++j) {
        // Current write position
        int write_pos = new_nnz;

        // Count non-zeros in the current column
        int non_zeros_in_col = 0;

        int start_idx = A_col_ptr[j];
        int end_idx = A_col_ptr[j + 1];

        for (int i = start_idx; i < end_idx; ++i) {
            if (A_values[i] != 0.0) {
                // In-place compaction of values and indices
                A_values[write_pos] = A_values[i];
                A_row_ind[write_pos] = A_row_ind[i];
                write_pos++;
                non_zeros_in_col++;
            }
        }
        new_nnz += non_zeros_in_col;
        temp_col_ptr[j + 1] = new_nnz;
    }

    // Resize the vectors to their new, smaller size
    A_values.resize(new_nnz);
    A_row_ind.resize(new_nnz);

    // Update the column pointer vector
    A_col_ptr = temp_col_ptr;
}

void BCSCtoCSR(
    const std::vector<double>& A_values,
    const std::vector<int>& A_row_ind,
    const std::vector<int>& A_col_ptr,
    int block_size,
    int num_rows,
    int num_cols,
    std::vector<double>& C_values,
    std::vector<int>& C_col_ind,
    std::vector<int>& C_row_ptr) {

    // Step 1: De-block and convert to a list of triplets (COO format)
    std::vector<Triplet> triplets;

    // A counter to keep track of the current block index in A_values
    int block_idx_counter = 0;

    // Loop through the BCSC columns (which are block columns)
    int num_block_cols = A_col_ptr.size() - 1;
    for (int j = 0; j < num_block_cols; ++j) {
        int start_block = A_col_ptr[j];
        int end_block = A_col_ptr[j + 1];

        // Loop through the blocks in the current block column
        for (int i = start_block; i < end_block; ++i) {
            int block_row_start = A_row_ind[i] * block_size;
            int block_col_start = j * block_size;

            // Iterate through the elements of the dense block
            for (int r = 0; r < block_size; ++r) {
                for (int c = 0; c < block_size; ++c) {
                    // Check for potential out-of-bounds access
                    if (block_idx_counter >= A_values.size()) {
                        std::cerr << "Error: BCSC index out of bounds." << std::endl;
                        return;
                    }

                    double val = A_values[block_idx_counter];
                    block_idx_counter++;

                    if (val != 0.0) {
                        triplets.push_back({block_row_start + r, block_col_start + c, val});
                    }
                }
            }
        }
    }

    // The rest of the code is correct, assuming the triplet list is built properly
    // Step 2: Sort the triplets by row index, then column index
    std::sort(triplets.begin(), triplets.end(), compareTriplets);

    // Step 3: Convert the sorted triplets to CSR format
    C_values.resize(triplets.size());
    C_col_ind.resize(triplets.size());
    C_row_ptr.assign(num_rows + 1, 0);

    for (size_t i = 0; i < triplets.size(); ++i) {
        C_values[i] = triplets[i].val;
        C_col_ind[i] = triplets[i].col;
        C_row_ptr[triplets[i].row + 1]++;
    }

    // Step 4: Convert the row pointer counts to cumulative sums
    for (int i = 1; i <= num_rows; ++i) {
        C_row_ptr[i] += C_row_ptr[i - 1];
    }
}

void BlockedCustomtoCSR(
    const std::vector<double>& A_values,
    const std::vector<int>& A_row_ind,
    const std::vector<int>& A_col_ptr,
    int block_size,
    int num_rows,
    int num_cols,
    std::vector<double>& C_values,
    std::vector<int>& C_col_ind,
    std::vector<int>& C_row_ptr) {

    std::vector<Triplet> triplets;

    int num_block_cols = A_col_ptr.size() - 1;

    // The index for the Drows vector, which holds the row indices for each element
    int Drows_idx = 0;

    for (int j_block = 0; j_block < num_block_cols; ++j_block) {
        int start_idx = A_col_ptr[j_block];
        int end_idx = A_col_ptr[j_block + 1];

        // The column for the current block
        int col_block_start = j_block * block_size;

        // Loop over the values within this block column
        for (int i = start_idx; i < end_idx; ++i) {
            // Reconstruct the global row index from the block structure
            int global_row_block = A_row_ind[i] / block_size;
            int local_row = A_row_ind[i] % block_size;
            int global_row = global_row_block * block_size + local_row;

            double val = A_values[i];

            if (val != 0.0) {
                triplets.push_back({global_row, col_block_start, val});
            }
        }
    }

    // Sort the triplets by row and column
    std::sort(triplets.begin(), triplets.end(), compareTriplets);

    C_values.resize(triplets.size());
    C_col_ind.resize(triplets.size());
    C_row_ptr.assign(num_rows + 1, 0);

    for (size_t i = 0; i < triplets.size(); ++i) {
        C_values[i] = triplets[i].val;
        C_col_ind[i] = triplets[i].col;

        if (triplets[i].row + 1 < C_row_ptr.size()) {
            C_row_ptr[triplets[i].row + 1]++;
        } else {
            std::cerr << "Error: Row index out of bounds for C_row_ptr." << std::endl;
            return;
        }
    }

    // Convert counts to cumulative sums
    for (int i = 1; i <= num_rows; ++i) {
        C_row_ptr[i] += C_row_ptr[i - 1];
    }
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

void constructCSRvals(
    const std::vector<double>& Cval,
    const std::vector<int>& Bcol_ind,
    const std::vector<int>& Brow_ptr,
    int Br, int Bc,
    std::vector<double>& Cval_csr) {

    int num_br = Brow_ptr.size() - 1;
    int num_bc = *std::max_element(Bcol_ind.begin(), Bcol_ind.end()) + 1;
    int M = num_br * Br;
    int N = num_bc * Bc;

    int csr_idx = 0;

    for (int I = 0; I < num_br; I++) {
        int block_start = Brow_ptr[I];
        int block_end = Brow_ptr[I + 1];
        for (int r = 0; r < Br; r++) {
            int i = I * Br + r;
            if (i >= M) break;
            for (int block_idx = block_start; block_idx < block_end; block_idx++) {
                int J = Bcol_ind[block_idx];
                for (int c = 0; c < Bc; c++) {
                    int j = J * Bc + c;
                    if (j >= N) continue;
                    int block_element_idx = block_idx * Br * Bc + r * Bc + c;
                    double value = Cval[block_element_idx];
                    if (value != 0) {
                        Cval_csr.push_back(value);
                        csr_idx++;
                    }
                }
            }
        }
    }
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

    int M = B_m * block_m;

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

    //printVector(y, "y CPU");

    // std::cout << B_m << " x " << B_n << std::endl;

    // std::cout << Bvals.size() << std::endl;
    // std::cout << Bcols.size() << std::endl;
    // std::cout << Brows.size() << std::endl;

    // printVector(Cvals, "Cvals");
    // printVector(Bvals, "Bvals");
    // printVector(Bcols, "Bcols");
    // printVector(Brows, "Brows");

    // std::cout << "##################################################################" << std::endl;

    // std::vector<double> csrBvals;
    // std::vector<int> csrBcols;
    // std::vector<int> csrBrows;

    // std::vector<double> csrCvals;
    // std::vector<int> csrCcols;
    // std::vector<int> csrCrows;

    // //CustomtoCSR(Dvals, Drows, Dcols, csrDvals, csrDcols, csrDrows);
    // // BlockedCustomtoCSR(Dvals, Drows, Dcols, block_m, D_m, D_n, csrDvals, csrDcols, csrDrows);

    // BCSRrecttoCSR(Bvals, Bcols, Brows, block_m, block_n, csrBvals, csrBcols, csrBrows);

    // BCSRrecttoCSR(Cvals, Bcols, Brows, block_m, block_n, csrCvals, csrCcols, csrCrows);

    // printVector(csrBvals, "csrBvals");
    // printVector(csrBcols, "csrBcols");
    // printVector(csrBrows, "csrBrows");
    // printVector(csrCvals, "csrCvals");
    // printVector(csrCcols, "csrCcols");
    // printVector(csrCrows, "csrCrows");

    umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    umfpack_di_free_numeric(&UMFPACK_Numeric);

    return 0;
}
