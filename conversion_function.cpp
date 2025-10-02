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
