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



        /*13,14,15, 16,17,18, 19,20,21, 22,23,24, 25,26,27, 28,29,30, 31,32,33, 34,35,36};*/

    // unsigned char s1[] = {1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3};
    // unsigned char s2[] = {4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6};
    // unsigned char s3[] = {7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9};

    int channel = 2;
    int height = 4;
    int width = 3;
    int batchSize = 1;

    // test data
    unsigned char s0[] = {1,12,  2,11,  3,10,
                          4,9,  5,8,  6,7,
                          7,6,  8,5,  9,4,
                          10,3, 11,2, 12,1};


    unsigned char s1[] = {3,11, 5,2, 4,8,
                          7,12, 9,5, 6,1,
                          6,11, 8,2, 9,4,
                          4,6,  6,8, 2,11};

    unsigned char s2[] = {7,7, 2,5, 3,10,
                          6,3, 9,8, 8,4,
                          3,2, 4,8, 5,9,
                          9,3, 1,5, 8,7};

    // Configure, initialize network
    KyuNet net;
    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);

    InitializerType weightInitType = InitializerType::ONE;
    InitializerType biasInitType = InitializerType::ZERO;
    Layer* pDenseLayer = net.addDenseLayer(pInputLayer, "Layer1", 4, ActivationType::RELU, weightInitType, biasInitType);
    net.Compile();


    int denseLayerSrcSize = pInputLayer->pDst_->getSize3D();
    fprintf(stderr, "denseLayerSrcSize: %d\n", denseLayerSrcSize);

    // Load data to input layer
    for (int i = 0; i < batchSize; ++i)
    {
        switch (i) {
            case 0:
            pInputLayer->FlattenImageToTensor(s0, false);
            break;
            case 1:
            pInputLayer->FlattenImageToTensor(s1, false);
            break;
            case 2:
            pInputLayer->FlattenImageToTensor(s2, false);
            break;
        }
    }

    // Display Dense layer information
    int weightsize3D = pDenseLayer->pW_->getSize3D();
    int weightWholeSize = pDenseLayer->pW_->getWholeSize();
    if (pDenseLayer->Type() == LayerType::FullConnected)
    {
        fprintf(stderr, "Dense Layer type is correct\n");
    }

    fprintf(stderr, "Dense Layer weight init type: \n");
    switch (pDenseLayer->Weight_Init_Type())
    {
        case InitializerType::NONE:
            fprintf(stderr, "NONE\n");
            break;
        case InitializerType::ZERO:
            fprintf(stderr, "ZERO\n");
            break;
        case InitializerType::TEST:
            fprintf(stderr, "TEST\n");
            break;
        case InitializerType::XAVIER_NORM:
            fprintf(stderr, "XAVIER\n");
            break;
        default:
            fprintf(stderr, "Default !!!\n");
            break;
    }
    fprintf(stderr, "Dense Layer bias init type: \n");
    switch (pDenseLayer->Bias_Init_Type())
    {
        case InitializerType::NONE:
            fprintf(stderr, "NONE\n");
            break;
        case InitializerType::ZERO:
            fprintf(stderr, "ZERO\n");
            break;
        case InitializerType::TEST:
            fprintf(stderr, "TEST\n");
            break;
        case InitializerType::XAVIER_NORM:
            fprintf(stderr, "XAVIER\n");
            break;
        default:
            fprintf(stderr, "Default !!!\n");
            break;
    }

    fprintf(stderr, "Dense Layer activation type: \n");
    ActivationType actType = pDenseLayer->Activation_Type();
    switch (actType) {
        case ActivationType::NONE:
            fprintf(stderr, "NONE\n");
            break;
        case ActivationType::SIGMOID:
            fprintf(stderr, "Sigmoid\n");
            break;
        case ActivationType::RELU:
            fprintf(stderr, "Relu\n");
            break;
        default:
            fprintf(stderr, "Default !!!\n");
            break;

    }
    fprintf(stderr, "\n");
    fprintf(stderr, "dense weight size3D: %d\n", weightsize3D);
    fprintf(stderr, "dense weight WholeSize: %d\n", weightWholeSize);



    // PRINT TEST RESULT
    fprintf(stderr, "Source of Dense Layer\n");
    float* pSrcData = pInputLayer->pDst_->getCPUData();
    // fprintf(stderr, "%.1f ", pSrcData[i]);
    for (int b = 0; b < batchSize; ++b)
    {
        fprintf(stderr, "[batch]: %d\n", b);
        for (int c = 0; c < channel; ++c)
        {
            fprintf(stderr, "  ch: %d\n", c);
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    fprintf(stderr, "\t%.2f\t", pSrcData[w + h*width + c*height*width + b*width*height*channel]);
                }
                fprintf(stderr, "\n");
            }
            // fprintf(stderr, "\n");
        }
        // fprintf(stderr, "\n");
    }


    fprintf(stderr, "\n\n");
    fprintf(stderr, "Weight of Dense Layer\n");
    for (int i = 0; i < pDenseLayer->pW_->getWidth(); ++i)
    {
        fprintf(stderr, "%d   ", i);
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < pDenseLayer->pW_->getWholeSize(); ++i)
    {
        if (i > 0 && (i % pDenseLayer->pW_->getWidth() == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pWData = pDenseLayer->pW_->getCPUData();
        fprintf(stderr, "%.1f ", pWData[i]);
    }
    fprintf(stderr, "\n\n");
    fprintf(stderr, "Bias of Dense Layer\n");
    for (int i = 0; i < pDenseLayer->pB_->getWholeSize(); ++i)
    {
        if (i > 0 && (i % pDenseLayer->pB_->getWidth() == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pBData = pDenseLayer->pB_->getCPUData();
        fprintf(stderr, "%.1f ", pBData[i]);
    }
    fprintf(stderr, "\n\n");


    pDenseLayer->Forward();


    fprintf(stderr, "Dense Layer Forward Result: \n");
    int denseLayer_output_size = pDenseLayer->pDst_->getSize3D();
    for (int i = 0; i < pDenseLayer->pDst_->getWholeSize(); ++i)
    {
        if (i > 0 && (i % denseLayer_output_size == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pDstData = pDenseLayer->pDst_->getCPUData();
        fprintf(stderr, "%.1f\t", pDstData[i]);
    }
    fprintf(stderr, "\n\n");

    // int tt = 4*4 + 7*8 + 10*12 + 2*16 + 5*20 + 8*24 + 11*28;
    // fprintf(stderr, "%d\n", tt);

    return 0;
}
