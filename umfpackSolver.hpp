#ifndef UMFPACK_SOLVER
#define UMFPACK_SOLVER

#include "linearSystemData.hpp"

#include <suitesparse/umfpack.h>
#include <algorithm>

namespace WellSolver
{
    class UMFPACKSolver
    {
        public:
            LinearSystemData& data_;
            const int block_m_;
            const int block_n_;
            const int B_m;
            const int B_n;
            const int M;
            std::vector<double> y;
            void *UMFPACK_Symbolic, *UMFPACK_Numeric;
            std::vector<double> z1;
            std::vector<double> z2;


            UMFPACKSolver(LinearSystemData& data);

            ~UMFPACKSolver();

            void cpuBx();

            void solveSystem();

            void cpuCz();
    };
}

#endif
