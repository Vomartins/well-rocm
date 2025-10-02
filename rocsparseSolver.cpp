#include "rocsparseSolver.hpp"

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

namespace WellSolver
{
    RocSPARSESolver::RocSPARSESolver(ConversionData& convData)
    :convData_(convData),
    B_M(convData_.csrBrows.size() - 1),
    B_N(*std::max_element(convData_.csrBcols.begin(), convData_.csrBcols.end()) + 1),
    B_nnz(convData.csrBvals.size()),
    C_M(convData_.csrCrows.size() - 1),
    C_N(*std::max_element(convData_.csrCcols.begin(), convData_.csrCcols.end()) + 1),
    C_nnz(convData_.csrCvals.size()),
    rocM(convData_.csrDrows.size() - 1),
    rocN(rocM),
    lda(rocM),
    ldb(rocM),
    nnzs(convData_.csrDvals.size())
    {

        if (!convData_.csrDvals.empty())
        {
            std::cout << "RocSPARSESolver initialized with data (CSR)." << std::endl;
            std::cout << "Data check: Dvals size is " << convData_.csrDvals.size() << std::endl;
            std::cout << "Data check: Dcols size is " << convData_.csrDcols.size() << std::endl;
            std::cout << "Data check: Drows size is " << convData_.csrDrows.size() << std::endl;
            std::cout << "Block Dimensions of B_w: " << B_M << " x " << B_N << std::endl;
            printf("############################################################## \n");
        }
        dataAlloc();
        dataToDevice();
        analyseMatrix();
        numFactorization();

        HIP_CALL(hipMemset(d_z, 0, ldb*Nrhs*sizeof(double)));

        h_z1.resize(B_M);
        h_z2.resize(B_M);
        h_y.resize(convData_.data_.y.size());
    }

    RocSPARSESolver::~RocSPARSESolver()
    {
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
    }

    void RocSPARSESolver::dataAlloc()
    {
        HIP_CALL(hipMalloc(&d_Dvals, sizeof(double)*size(convData_.csrDvals)));
        checkHIPAlloc(d_Dvals);
        HIP_CALL(hipMalloc(&d_Dcols, sizeof(int)*size(convData_.csrDcols)));
        checkHIPAlloc(d_Dcols);
        HIP_CALL(hipMalloc(&d_Drows, sizeof(int)*size(convData_.csrDrows)));
        checkHIPAlloc(d_Drows);
        HIP_CALL(hipMalloc(&d_Bvals, sizeof(double)*size(convData_.csrBvals)));
        checkHIPAlloc(d_Bvals);
        HIP_CALL(hipMalloc(&d_Bcols, sizeof(int)*size(convData_.csrBcols)));
        checkHIPAlloc(d_Bcols);
        HIP_CALL(hipMalloc(&d_Brows, sizeof(int)*size(convData_.csrBrows)));
        checkHIPAlloc(d_Brows);
        HIP_CALL(hipMalloc(&d_Cvals, sizeof(double)*size(convData_.csrCvals)));
        checkHIPAlloc(d_Cvals);
        HIP_CALL(hipMalloc(&d_Ccols, sizeof(int)*size(convData_.csrCcols)));
        checkHIPAlloc(d_Ccols);
        HIP_CALL(hipMalloc(&d_Crows, sizeof(int)*size(convData_.csrCrows)));
        checkHIPAlloc(d_Crows);
        HIP_CALL(hipMalloc(&d_z, sizeof(double)*ldb*Nrhs));
        checkHIPAlloc(d_z);
        HIP_CALL(hipMalloc(&d_z_aux, sizeof(double)*ldb*Nrhs));
        checkHIPAlloc(d_z_aux);
        HIP_CALL(hipMalloc(&d_x, sizeof(double)*size(convData_.data_.x)));
        checkHIPAlloc(d_x);
        HIP_CALL(hipMalloc(&d_y, sizeof(double)*size(convData_.data_.y)));
        checkHIPAlloc(d_y);
    }

    void RocSPARSESolver::dataToDevice()
    {
        HIP_CALL(hipMemcpy(d_Dvals, convData_.csrDvals.data(), size(convData_.csrDvals)*sizeof(double), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_Dcols, convData_.csrDcols.data(), size(convData_.csrDcols)*sizeof(int), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_Drows, convData_.csrDrows.data(), size(convData_.csrDrows)*sizeof(int), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_Bvals, convData_.csrBvals.data(), size(convData_.csrBvals)*sizeof(double), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_Bcols, convData_.csrBcols.data(), size(convData_.csrBcols)*sizeof(int), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_Brows, convData_.csrBrows.data(), size(convData_.csrBrows)*sizeof(int), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_Cvals, convData_.csrCvals.data(), size(convData_.csrCvals)*sizeof(double), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_Ccols, convData_.csrCcols.data(), size(convData_.csrCcols)*sizeof(int), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_Crows, convData_.csrCrows.data(), size(convData_.csrCrows)*sizeof(int), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_x, convData_.data_.x.data(), size(convData_.data_.x)*sizeof(double), hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(d_y, convData_.data_.y.data(), size(convData_.data_.y)*sizeof(double), hipMemcpyHostToDevice));
    }

    void RocSPARSESolver::analyseMatrix()
    {
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
    }

    void RocSPARSESolver::numFactorization()
    {
        ROCSPARSE_CALL(rocsparse_dcsrilu0(handle, rocM, nnzs, descr_M,
				      d_Dvals, d_Drows, d_Dcols, ilu_info, rocsparse_solve_policy_auto, d_buffer));

        status = rocsparse_csrilu0_zero_pivot(handle, ilu_info, &zero_position);
        if (status != rocsparse_status_success) {
            printf("--- RocSPARSE Error --- L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
        }
    }

    void RocSPARSESolver::Bx()
    {
        double alpha = 1.0;
        double beta = 0.0;

        ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(handle, rocsparse_operation_none, B_M, B_N, B_nnz, descr_B, d_Bvals, d_Brows, d_Bcols, B_info));

        ROCSPARSE_CALL(rocsparse_dcsrmv(handle, rocsparse_operation_none, B_M, B_N, B_nnz, &alpha, descr_B, d_Bvals, d_Brows, d_Bcols, B_info, d_x, &beta, d_z));

        z1ToHost();

        HIP_CALL(hipGetLastError()); // Check for errors
        HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
    }

    void RocSPARSESolver::z1ToHost()
    {
        HIP_CALL(hipMemcpy(h_z1.data(), d_z, sizeof(double) * B_M, hipMemcpyDeviceToHost));
    }

    void RocSPARSESolver::solveDw()
    {
        ROCSPARSE_CALL(rocsparse_dcsrsv_solve(handle, \
                                  operation, rocM, nnzs, &one, \
					  descr_L, d_Dvals, d_Drows, d_Dcols,  ilu_info, d_z, d_z_aux, rocsparse_solve_policy_auto, d_buffer));
        ROCSPARSE_CALL(rocsparse_dcsrsv_solve(handle,\
                                  operation, rocM, nnzs, &one, \
					  descr_U, d_Dvals, d_Drows, d_Dcols, ilu_info, d_z_aux, d_z, rocsparse_solve_policy_auto, d_buffer));
        z2ToHost();
    }

    void RocSPARSESolver::z2ToHost()
    {
        HIP_CALL(hipMemcpy(h_z2.data(), d_z, sizeof(double) * B_M, hipMemcpyDeviceToHost));
    }

    void RocSPARSESolver::Cz()
    {
        double alpha = -1.0;
        double beta = 1.0;

        ROCSPARSE_CALL(rocsparse_dcsrmv_analysis(handle, rocsparse_operation_transpose, C_M, C_N, C_nnz, descr_C, d_Cvals, d_Crows, d_Ccols, C_info));

        ROCSPARSE_CALL(rocsparse_dcsrmv(handle, rocsparse_operation_transpose, C_M, C_N, C_nnz, &alpha, descr_C, d_Cvals, d_Crows, d_Ccols, C_info, d_z, &beta, d_y));

        yToHost();

        HIP_CALL(hipGetLastError()); // Check for errors
        HIP_CALL(hipDeviceSynchronize()); // Synchronize after kernel execution
    }

    void RocSPARSESolver::yToHost()
    {
        HIP_CALL(hipMemcpy(h_y.data(), d_y, sizeof(double) * B_M, hipMemcpyDeviceToHost));
    }
}
