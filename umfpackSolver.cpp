#include "umfpackSolver.hpp"

#include <suitesparse/umfpack.h>
#include <algorithm>

namespace WellSolver
{
    UMFPACKSolver::UMFPACKSolver(LinearSystemData& data)
    :data_(data),
     block_m_(data_.block_m_),
     block_n_(data_.block_n_),
     B_m(data_.Brows.size() - 1),
     B_n(*std::max_element(data_.Bcols.begin(), data_.Bcols.end()) + 1),
     M(B_m * block_m_),
     y(data_.y)
    {
        std::cout << "UMFPACLSolver initialized with data." << std::endl;

        if (!data_.Dvals.empty()) {
            std::cout << "Data check: Dvals size is " << data_.Dvals.size() << std::endl;
            std::cout << "Data check: Dcols size is " << data_.Dcols.size() << std::endl;
            std::cout << "Data check: Drows size is " << data_.Drows.size() << std::endl;
            std::cout << "Block Dimensions of B_w: " << B_m << " x " << B_n << std::endl;
            printf("############################################################## \n");
        }

        umfpack_di_symbolic(M, M, data_.Dcols.data(), data_.Drows.data(), data_.Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
        umfpack_di_numeric(data_.Dcols.data(), data_.Drows.data(), data_.Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);

        z1.resize(B_m * block_m_);
        z2.resize(B_m * block_m_);

        std::fill(z1.begin(), z1.end(), 0.0);
        std::fill(z2.begin(), z2.end(), 0.0);
    }

    UMFPACKSolver::~UMFPACKSolver()
    {
        if (UMFPACK_Symbolic) {
            umfpack_di_free_symbolic(&UMFPACK_Symbolic);
            // std::cout << "UMFPACK_Symbolic freed." << std::endl;
        }
        if (UMFPACK_Numeric) {
            umfpack_di_free_numeric(&UMFPACK_Numeric);
            // std::cout << "UMFPACK_Numeric freed." << std::endl;
        }
    }

    void UMFPACKSolver::cpuBx()
    {
        for (unsigned int row = 0; row < B_m; ++row) {
            // for every block in the row
            for (unsigned int blockID = data_.Brows[row]; blockID < data_.Brows[row + 1]; ++blockID) {
                unsigned int colIdx = data_.Bcols[blockID];
                for (unsigned int j = 0; j < block_m_; ++j) {
                    double temp = 0.0;
                    for (unsigned int k = 0; k < block_n_; ++k) {
                        temp += data_.Bvals[blockID * block_n_ * block_m_ + j * block_n_ + k] * data_.x[colIdx * block_n_ + k];
                    }
                    z1[row * block_m_ + j] += temp;
                }
            }
        }
    }

    void UMFPACKSolver::solveSystem()
    {
        umfpack_di_solve(UMFPACK_A, data_.Dcols.data(), data_.Drows.data(), data_.Dvals.data(), z2.data(), z1.data(), UMFPACK_Numeric, nullptr, nullptr);
    }

    void UMFPACKSolver::cpuCz()
    {
        for (unsigned int row = 0; row < B_m; ++row) {
            // for every block in the row
            for (unsigned int blockID = data_.Brows[row]; blockID < data_.Brows[row + 1]; ++blockID) {
                unsigned int colIdx = data_.Bcols[blockID];
                for (unsigned int j = 0; j < block_n_; ++j) {
                    double temp = 0.0;
                    for (unsigned int k = 0; k < block_m_; ++k) {
                        temp += data_.Cvals[blockID * block_n_ * block_m_ + j + k * block_n_] * z2[row * block_m_ + k];
                    }
                    y[colIdx * block_n_ + j] -= temp;
                }
            }
        }
    }
}
