#ifndef ROCSPARSE_SOLVER
#define ROCSPARSE_SOLVER

#include "conversionData.hpp"

#include <rocsparse/rocsparse.h>

namespace WellSolver
{
    class RocSPARSESolver
    {
        public:
            ConversionData& convData_;
            const int B_M;
            const int B_N;
            const int B_nnz;
            const int C_M;
            const int C_N;
            const int C_nnz;

            // RocSPARSE
            const double one  = 1.0;
            const rocsparse_int rocM;
            const rocsparse_int rocN;
            const rocsparse_int Nrhs = 1;
            const rocsparse_int lda;
            const rocsparse_int ldb;
            const rocsparse_int nnzs;
            rocsparse_mat_info ilu_info, B_info, C_info;
            rocsparse_mat_descr descr_M, descr_L, descr_U, descr_B, descr_C;
            std::size_t d_bufferSize_M, d_bufferSize_L, d_bufferSize_U, d_bufferSize;
            void *d_buffer;
            rocsparse_handle handle;
            rocsparse_operation operation = rocsparse_operation_none;
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

            std::vector<double> h_z1;
            std::vector<double> h_z2;
            std::vector<double> h_y;

            RocSPARSESolver(ConversionData& convData);

            ~RocSPARSESolver();

            void dataAlloc();

            void dataToDevice();

            void analyseMatrix();

            void numFactorization();

            void Bx();

            void z1ToHost();

            void solveDw();

            void z2ToHost();

            void Cz();

            void yToHost();




    };
}

#endif
