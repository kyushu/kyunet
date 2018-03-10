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
    unsigned char s0[] = {1,3,  2,4,  3,6,
                          4,7,  5,8,  3,7,
                          7,3,  8,7,  9,2,
                          10,5, 11,7, 12,10};


    unsigned char s1[] = {3,11, 5,2, 4,8,
                          7,12, 9,5, 6,1,
                          6,11, 8,2, 9,4,
                          4,6,  6,8, 2,11};

    unsigned char s2[] = {7,7, 2,5, 3,10,
                          6,3, 9,8, 8,4,
                          3,2, 4,8, 5,9,
                          9,3, 1,5, 8,7};

    // Configure, initialize network
    Net net;
    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);

    InitializerType weightInitType = InitializerType::ONE;
    InitializerType biasInitType = InitializerType::ZERO;
    Layer* pDenseLayer = net.addDenseLayer(pInputLayer, "Layer1", 4, ActivationType::Relu, weightInitType, biasInitType);
    net.initialize();

    // Get InputLayer
    // InputLayer* pInput = net.getInputLayer();
    // const float *pdata = pInput->pDst->cpu_data();
    int denseLayerSrcSize = pDenseLayer->pSrc_->Size3D();
    fprintf(stderr, "denseLayerSrcSize: %d\n", denseLayerSrcSize);

    // Add data
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

    // pInput->FlattenImageToTensor(s1, false);
    // pInput->FlattenImageToTensor(s2, true);
    // pInput->FlattenImageToTensor(s3, true);

    int weightsize3D = pDenseLayer->pW_->Size3D();
    int weightWholeSize = pDenseLayer->pW_->WholeSize();
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
        case ActivationType::Sigmoid:
            fprintf(stderr, "Sigmoid\n");
            break;
        case ActivationType::Tanh:
            fprintf(stderr, "Tanh\n");
            break;
        case ActivationType::Relu:
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
    float* pSrcData = pDenseLayer->pSrc_->cpu_data();
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
    for (int i = 0; i < pDenseLayer->pW_->WholeSize(); ++i)
    {
        if (i > 0 && (i % pDenseLayer->pW_->Width() == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pWData = pDenseLayer->pW_->cpu_data();
        fprintf(stderr, "%.1f ", pWData[i]);
    }
    fprintf(stderr, "\n\n");
    fprintf(stderr, "Bias of Dense Layer\n");
    for (int i = 0; i < pDenseLayer->pB_->WholeSize(); ++i)
    {
        if (i > 0 && (i % pDenseLayer->pB_->Width() == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pBData = pDenseLayer->pB_->cpu_data();
        fprintf(stderr, "%.1f ", pBData[i]);
    }
    fprintf(stderr, "\n\n");


    pDenseLayer->Forward();


    fprintf(stderr, "Dense Layer Forward Result: \n");
    int denseLayer_output_size = pDenseLayer->pDst_->Size3D();
    for (int i = 0; i < pDenseLayer->pDst_->WholeSize(); ++i)
    {
        if (i > 0 && (i % denseLayer_output_size == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pDstData = pDenseLayer->pDst_->cpu_data();
        fprintf(stderr, "%.1f\t", pDstData[i]);
    }
    fprintf(stderr, "\n\n");

    // int tt = 4*4 + 7*8 + 10*12 + 2*16 + 5*20 + 8*24 + 11*28;
    // fprintf(stderr, "%d\n", tt);

    return 0;
}
