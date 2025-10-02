
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
#include "error.hpp"

#include <dune/common/timer.hh>

#include <Eigen/Dense>
#include <limits> // For infinity()

int main(int argc, char ** argv)
{
    int block_m = 4;
    int block_n = 3;

    // Linear system data load
    //
    WellSolver::LinearSystemData data(block_m, block_n);

    data.printDataSizes();

    // UMFPACK solver initiation and run

    WellSolver::UMFPACKSolver umfpackSolver(data);

    umfpackSolver.cpuBx();

    umfpackSolver.solveSystem();

    umfpackSolver.cpuCz();

    // Data sparse storage format convertion

    WellSolver::ConversionData convData(data);

    convData.CustomtoCSR();

    convData.ConvertB();

    convData.ConvertC();

    convData.printDataSizes();

    // RocSPARSE solver initiation and run

    WellSolver::RocSPARSESolver rocsparseSolver(convData);

    rocsparseSolver.Bx();

    rocsparseSolver.solveDw();

    rocsparseSolver.Cz();

    WellSolver::ErrorReport errorReport(umfpackSolver, rocsparseSolver, convData);

    return 0;
}
