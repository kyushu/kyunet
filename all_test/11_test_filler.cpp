#include "filler.hpp"
#include "tensor.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    mkt::Tensor<float> tensor{1, 1, 100, 100};
    tensor.allocate();
    Xavier<float> xavier{Distribution::UNIFORM};
    xavier(tensor);

    // Debug Display
    float* pData = tensor.getCPUData();
    float sum = 0;
    for (int i = 0; i < tensor.getWholeSize(); ++i)
    {
        sum += pData[i];
        // fprintf(stderr, "d[%d]=%f\n", i, pData[i]);
    }

    float avg = sum / tensor.getWholeSize();
    fprintf(stderr, "avg= %f\n", avg);

    return 0;
}
