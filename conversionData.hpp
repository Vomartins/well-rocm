#ifndef CONVERSION_DATA
#define CONVERSION_DATA

#include "linearSystemData.hpp"

#include<algorithm>
#include<numeric>

namespace WellSolver
{
    class ConversionData
    {
        public:
            LinearSystemData& data_;
            const int block_m_;
            const int block_n_;

            std::vector<double> csrBvals;
            std::vector<int> csrBcols;
            std::vector<int> csrBrows;

            std::vector<double> csrCvals;
            std::vector<int> csrCcols;
            std::vector<int> csrCrows;

            std::vector<double> csrDvals;
            std::vector<int> csrDcols;
            std::vector<int> csrDrows;

            ConversionData(LinearSystemData& data);

            void CustomtoCSR();

            void BCSRrecttoCSR(std::vector<double>& Bval,
                                std::vector<int>& Bcol_ind,
                                std::vector<int>& Brow_ptr,
                                int Br, int Bc,
                                std::vector<double>& val,
                                std::vector<int>& col_ind,
                                std::vector<int>& row_ptr);

            void ConvertB();

            void ConvertC();
    };
}
#endif
