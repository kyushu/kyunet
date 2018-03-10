
#include <cstdlib>

#include "net.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    // Net Configuration
    int batchSize = 1;
    int height = 4;
    int width = 4;
    int channel = 3;

    Net net;

    // Add Input Lauer
    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);
    // Add Rely Layer
    Layer* pSigmoidLayer = net.addSigmoidLayer(pInputLayer, "ReluLayer");

    // Net Initialization: allocation memory space for each Tensor of layer
    net.initialize();

    // Feed pseudo data
    Tensor* pInputTensor = pInputLayer->pDst_;
    float* pInDstData = pInputTensor->cpu_data();
    fprintf(stderr, "Input data\n");
    for (int i = 0; i < pInputTensor->WholeSize(); ++i)
    {
        float fval = (float(std::rand() % 100) / 100 - 0.5);
        pInDstData[i] = fval;
        fprintf(stderr, "%d: %f\n",i, fval);
    }

    pSigmoidLayer->Forward();

    int wholeSize = pSigmoidLayer->pDst_->WholeSize();
    float* pDstData = pSigmoidLayer->pDst_->cpu_data();
    fprintf(stderr, "Relu result\n");
    for (int i = 0; i < wholeSize; ++i)
    {
        fprintf(stderr, "src[%d]: %f, dst[%d]: %f\n", i, pInDstData[i], i, pDstData[i]);
    }
    return 0;
}
