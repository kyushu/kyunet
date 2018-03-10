#include "filler.hpp"
#include "tensor.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    mkt::Tensor tensor{1, 100, 100, 1};
    tensor.allocate();
    Xavier xavier{Distribution::UNIFORM};
    xavier(tensor);

    // Debug Display
    float* pData = tensor.cpu_data();
    float sum = 0;
    for (int i = 0; i < tensor.WholeSize(); ++i)
    {
        sum += pData[i];
        // fprintf(stderr, "d[%d]=%f\n", i, pData[i]);
    }

    float avg = sum / tensor.WholeSize();
    fprintf(stderr, "avg= %f\n", avg);

    return 0;
}
