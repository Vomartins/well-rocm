#include "conversionData.hpp"

bool compareTuple(const std::tuple<double, int, int>& a, const std::tuple<double, int, int>& b) {
    if (std::get<1>(a) != std::get<1>(b)) {
        return std::get<1>(a) < std::get<1>(b);
    }
    return std::get<2>(a) < std::get<2>(b);
}

namespace WellSolver
{
    ConversionData::ConversionData(LinearSystemData& data)
    :data_(data),
     block_m_(data_.block_m_),
     block_n_(data_.block_n_)
    {
         if (!data_.Dvals.empty()) {
             printf("Data converted to CSR format. \n");
             printf("############################################################## \n");
         }

    }

    void ConversionData::CustomtoCSR()
    {
        int D_m = data_.Dcols.size() - 1;
        std::vector<int> Cols(data_.Dvals.size());

        for (int i=0; i< data_.Dcols.size()-1; i++){
            for (int j=data_.Dcols[i]; j<data_.Dcols[i+1]; j++){
                Cols[j] = i;
            }
        }

        std::vector<std::tuple<double, int, int>> COO;

        for (int i=0; i<data_.Dvals.size(); i++){
            COO.push_back(std::make_tuple(data_.Dvals[i], data_.Drows[i], Cols[i]));
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

    void ConversionData::BCSRrecttoCSR(
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

    void ConversionData::ConvertB()
    {
        BCSRrecttoCSR(data_.Bvals, data_.Bcols, data_.Brows, block_m_, block_n_, csrBvals, csrBcols, csrBrows);
    }

    void ConversionData::ConvertC()
    {
        BCSRrecttoCSR(data_.Cvals, data_.Bcols, data_.Brows, block_m_, block_n_, csrCvals, csrCcols, csrCrows);
    }
}
