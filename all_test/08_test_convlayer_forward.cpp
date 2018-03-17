
#include <iostream>
#include <cstdio>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.hpp"
#include "folder_file_utils.hpp"

#include "tensor.h"
#include "layer/layer.h"
#include "layer/input_layer.h"
#include "net.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    // Net Configuration
    int batchSize = 1;
    int height = 9;
    int width = 9;
    int channel = 1;

    Net net;
    // Add Input Layer
    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);

    // Add ConvLayer
    InitializerType weightInitType = InitializerType::TEST;
    InitializerType biasInitType = InitializerType::ZERO;
    Layer* pConvLayer = net.addConvLayer(pInputLayer, "conv_1", 3, 3, 2, 1, 1, 0, 0, PaddingType::VALID, ActivationType::NONE, weightInitType, biasInitType);


    // Net Initialization: memory allocation
    net.initialize();

    // Set pseudo data
    Tensor* pInputTensor = pInputLayer->pDst_;
    float* pInDstData = pInputTensor->cpu_data();
    for (int i = 0; i < pInputTensor->WholeSize(); ++i)
    {
        float fval = (float(std::rand() % 10));
        pInDstData[i] = fval;
    }

    // input
    int ih = pInputTensor->Height();
    int iw = pInputTensor->Width();
    int ic = pInputTensor->Channel();
    fprintf(stderr, "Input\n");
    for (int b = 0; b < batchSize; ++b)
    {
        for (int c = 0; c < ic; ++c)
        {
            for (int h = 0; h < ih; ++h)
            {
                for (int w = 0; w < iw; ++w)
                {
                    fprintf(stderr, "[%d] = %.3f\t", w + h*iw + c*ih*iw + b*ic*ih*iw, pInDstData[w + h*iw + c*ih*iw + b*ic*ih*iw]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }

    pConvLayer->Forward();

    // weight
   Tensor *pW = pConvLayer->pW_;
   float* pWData = pW->cpu_data();
   int fh = pW->Height();
   int fw = pW->Width();
   int fc = pW->Channel();
   fprintf(stderr, "Weight\n");
   for (int c = 0; c < fc; ++c)
   {
       for (int h = 0; h < fh; ++h)
       {
           for (int w = 0; w < fw; ++w)
           {
               fprintf(stderr, "%.2f\t", pWData[w + h*fw + c*fh*fw]);
           }
           fprintf(stderr, "\n");
       }
       fprintf(stderr, "\n");
   }
   fprintf(stderr, "\n");

    // output
    Tensor *pDst = pConvLayer->pDst_;
    float* pDstData = pDst->cpu_data();
    int batchsize = pDst->NumOfData();
    int wholeSize = pDst->WholeSize();
    int oh = pDst->Height();
    int ow = pDst->Width();
    int oc = pDst->Channel();
    fprintf(stderr, "Output\n");
    for (int b = 0; b < batchsize; ++b)
    {
        for (int c = 0; c < oc; ++c)
        {
            for (int h = 0; h < oh; ++h)
            {
                for (int w = 0; w < ow; ++w)
                {
                    fprintf(stderr, "[%d]=%.3f\t", w + h*ow + c*oh*ow + b*oc*oh*ow, pDstData[w + h*ow + c*oh*ow + b*oc*oh*ow]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
    }


    return 0;
}
