#include "linearSystemData.hpp"

namespace WellSolver
{
    LinearSystemData::LinearSystemData(const int block_m, const int block_n)
    :block_m_(block_m),
     block_n_(block_n)
    {
        readVector(Dvals, "data/Dvals.txt");
        readVector(Dcols, "data/Dcols.txt");
        readVector(Drows, "data/Drows.txt");
        readVector(Bvals, "data/Bvals.txt");
        readVector(Bcols, "data/Bcols.txt");
        readVector(Brows, "data/Brows.txt");
        readVector(Cvals, "data/Cvals.txt");
        readVector(x, "data/vecx.txt");
        readVector(y, "data/vecy.txt");
    }

    void LinearSystemData::printDataSizes()
    {
        printf("######################### Data sizes ######################### \n");
        printf("Vector x size : %zu \n", x.size());
        printf("Vector y size : %zu \n", y.size());
        printf("Dvals size : %zu \n", Dvals.size());
        printf("Dcols size : %zu \n", Dcols.size());
        printf("Drows size : %zu \n", Drows.size());
        printf("Bvals size : %zu \n", Bvals.size());
        printf("Bcols size : %zu \n", Bcols.size());
        printf("Brows size : %zu \n", Brows.size());
        printf("Cvals size : %zu \n", Cvals.size());
        printf("############################################################## \n");
    }
}
