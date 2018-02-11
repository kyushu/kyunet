
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
    Layer* pReluLayer = net.addReluLayer(pInputLayer, "ReluLayer");

    // Net Initialization: allocation memory space for each Tensor of layer
    net.initialize();

    // Feed pseudo data
    Tensor* pInputTensor = pInputLayer->pDst_;
    float* pInDstData = pInputTensor->getData();
    fprintf(stderr, "Input data\n");
    for (int i = 0; i < pInputTensor->getWholeSize(); ++i)
    {
        float fval = (float(std::rand() % 100) / 100 - 0.5);
        pInDstData[i] = fval;
        fprintf(stderr, "%d: %f\n",i, fval);
    }

    pReluLayer->forward();

    int wholeSize = pReluLayer->pDst_->getWholeSize();
    float* pDstData = pReluLayer->pDst_->getData();
    fprintf(stderr, "Relu result\n");
    for (int i = 0; i < wholeSize; ++i)
    {
        fprintf(stderr, "src[%d]: %f, dst[%d]: %f\n", i, pInDstData[i], i, pDstData[i]);
    }
    return 0;
}