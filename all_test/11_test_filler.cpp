#include "filler.h"
#include "tensor.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    mkt::Tensor tensor{1, 100, 100, 1};
    tensor.allocate();
    Xavier xavier{Distribution::UNIFORM};
    // Xavier xavier{};
    xavier(tensor);

    float* pData = tensor.getData();
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
