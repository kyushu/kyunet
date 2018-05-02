
#include <cstdlib>

#include "net.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    // KyuNet Configuration
    int batchSize = 1;
    int height = 4;
    int width = 4;
    int channel = 3;

    KyuNet net;

    // Add Input Lauer
    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);
    // Add Rely Layer
    Layer* pSigmoidLayer = net.addSigmoidLayer(pInputLayer, "ReluLayer");

    // KyuNet Initialization: allocation memory space for each Tensor of layer
    net.Compile();

    // Feed pseudo data
    Tensor* pInputTensor = pInputLayer->pDst_;
    float* pInDstData = pInputTensor->getCPUData();
    fprintf(stderr, "Input data\n");
    for (int i = 0; i < pInputTensor->getWholeSize(); ++i)
    {
        float fval = (float(std::rand() % 100) / 100 - 0.5);
        pInDstData[i] = fval;
        fprintf(stderr, "%d: %f\n",i, fval);
    }

    pSigmoidLayer->Forward();

    int wholeSize = pSigmoidLayer->pDst_->getWholeSize();
    float* pDstData = pSigmoidLayer->pDst_->getCPUData();
    fprintf(stderr, "Relu result\n");
    for (int i = 0; i < wholeSize; ++i)
    {
        fprintf(stderr, "src[%d]: %f, dst[%d]: %f\n", i, pInDstData[i], i, pDstData[i]);
    }
    return 0;
}
