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
}
