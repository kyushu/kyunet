
#include <iostream>
#include <cstdio>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.hpp"
#include "folder_file_utils.hpp"

#include "tensor.h"
#include "layer.h"
#include "inputLayer.h"
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
    InputLayer* pInputLyaer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);

    // Add ConvLayer
    InitializerType weightInitType = InitializerType::ONE;
    InitializerType biasInitType = InitializerType::ONE;
    Layer* pConvLayer = net.addConvLayer(pInputLyaer, "conv_1", 2, 3, 1, 0, PaddingType::valid, ActivationType::NONE, weightInitType, biasInitType);


    // Net Initialization: memory allocation
    net.initialize();

    // Set pseudo data
    Tensor* pInputTensor = pInputLyaer->pDst_;
    float* pInDstData = pInputTensor->getData();
    for (int i = 0; i < pInputTensor->getWholeSize(); ++i)
    {
        pInDstData[i] = i;
    }

    pConvLayer->forward();

    Tensor *pDst = pConvLayer->pDst_;
    float* pDstData = pDst->getData();
    for (int i = 0; i < pDst->getWholeSize(); ++i)
    {
         if (i > 0 && i%pDst->getSize2D()==0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%f ", pDstData[i]);
    }
    mktLog(2, "\n");


    return 0;
}
