
#include "conversionData.hpp"
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include <math.h>
#include <vector>
#include <string>

#include "linearSystemData.hpp"
#include "umfpackSolver.hpp"
#include "conversionData.hpp"
#include "rocsparseSolver.hpp"

#include <dune/common/timer.hh>




#include <Eigen/Dense>
#include <limits> // For infinity()


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

    WellSolver::RocSPARSESolver rocsparseSolver(convData);

    rocsparseSolver.Bx();

    // rocsparseSolver.z1ToHost();

    rocsparseSolver.solveDw();

    // rocsparseSolver.z2ToHost();

    rocsparseSolver.Cz();

    // rocsparseSolver.yToHost();

    double rel_err;
    printf("--- Relative error h_z1 and z1 --- \n");
    rel_err = relativeErrorInfinityNorm(rocsparseSolver.h_z1, umfpackSolver.z1);
    printf("rel_error = %f \n", rel_err);

    printf("--- Relative error h_z2 and z2 --- \n");
    rel_err = relativeErrorInfinityNorm(rocsparseSolver.h_z2, umfpackSolver.z2);
    printf("rel_error = %f \n", rel_err);

    printf("--- Relative error h_y and y --- \n");
    rel_err = relativeErrorInfinityNorm(rocsparseSolver.h_y, umfpackSolver.y);
    printf("rel_error = %f \n", rel_err);

    std::vector<double> Dz;
    Dz.resize(umfpackSolver.z1.size());

    printf("--- Infinity norm of residual of D_w z = B_w x (RocSPARSE) --- \n");
    spmv_csr(convData.csrDvals, convData.csrDcols, convData.csrDrows, rocsparseSolver.h_z2, Dz, convData.csrDrows.size() - 1);
    rel_err = relativeErrorInfinityNorm(Dz, rocsparseSolver.h_z1);
    printf("|Dz-Bx| = %f \n", rel_err);

    printf("--- Infinity norm of residual of D_w z = B_w x (UMFPACK) --- \n");
    spmv_csr(convData.csrDvals, convData.csrDcols, convData.csrDrows, umfpackSolver.z2, Dz, convData.csrDrows.size() - 1);
    rel_err = relativeErrorInfinityNorm(Dz, umfpackSolver.z1);
    printf("|Dz-Bx| = %f \n", rel_err);

    return 0;
}
