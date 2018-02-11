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

    // Preset test data
    unsigned char s0[] = {1,2, 4,5, 7,8, 10,11};
    unsigned char s1[] = {3,5, 7,9, 6,8, 4,6};
    unsigned char s2[] = {7,2, 6,9, 3,4, 9,1};

        /*13,14,15, 16,17,18, 19,20,21, 22,23,24, 25,26,27, 28,29,30, 31,32,33, 34,35,36};*/

    // unsigned char s1[] = {1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3};
    // unsigned char s2[] = {4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6};
    // unsigned char s3[] = {7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9};

    int channel = 2/*3*/;
    int height = 2/*4*/;
    int width = 2/*3*/;
    int batchSize = 3;

    // Configure, initialize network
    Net net;
    InputLayer* pInputLyaer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);

    InitializerType weightInitType = InitializerType::TEST;
    InitializerType biasInitType = InitializerType::ZERO;
    Layer* pDenseLayer = net.addDenseLayer(pInputLyaer, "Layer1", 4, ActivationType::Relu, weightInitType, biasInitType);
    net.initialize();

    // Get InputLayer
    // InputLayer* pInput = net.getInputLayer();
    // const float *pdata = pInput->pDst->getData();
    int denseLayerSrcSize = pDenseLayer->pSrc_->getSize3D();
    fprintf(stderr, "denseLayerSrcSize: %d\n", denseLayerSrcSize);

    // Add data
    pInputLyaer->FlattenImageToTensor(s0, false);
    pInputLyaer->FlattenImageToTensor(s1, false);
    pInputLyaer->FlattenImageToTensor(s2, false);

    // pInput->FlattenImageToTensor(s1, false);
    // pInput->FlattenImageToTensor(s2, true);
    // pInput->FlattenImageToTensor(s3, true);

    int weightsize3D = pDenseLayer->pW_->getSize3D();
    int weightWholeSize = pDenseLayer->pW_->getWholeSize();
    if (pDenseLayer->getType() == LayerType::FullConnected)
    {
        fprintf(stderr, "Dense Layer type is correct\n");
    }

    fprintf(stderr, "Dense Layer weight init type: \n");
    switch (pDenseLayer->getWeightInitType())
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
    switch (pDenseLayer->getBiasInitType())
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
    ActivationType actType = pDenseLayer->getActivationType();
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
    for (int i = 0; i < pDenseLayer->pSrc_->getWholeSize(); ++i)
    {
        if (i > 0 && (i % pDenseLayer->pSrc_->getSize3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pSrcData = pDenseLayer->pSrc_->getData();
        fprintf(stderr, "%.1f ", pSrcData[i]);
    }

    fprintf(stderr, "\n\n");
    fprintf(stderr, "Weight of Dense Layer\n");
    for (int i = 0; i < pDenseLayer->pW_->getWholeSize(); ++i)
    {
        if (i > 0 && (i % pDenseLayer->pW_->getWidth() == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pWData = pDenseLayer->pW_->getData();
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
        float* pBData = pDenseLayer->pB_->getData();
        fprintf(stderr, "%.1f ", pBData[i]);
    }
    fprintf(stderr, "\n\n");


    pDenseLayer->forward();
    fprintf(stderr, "Dense Layer Forward Result: \n");
    int denseLayer_output_size = pDenseLayer->pDst_->getSize3D();
    for (int i = 0; i < pDenseLayer->pDst_->getWholeSize(); ++i)
    {
        if (i > 0 && (i % denseLayer_output_size == 0))
        {
            fprintf(stderr, "\n");
        }
        float* pDstData = pDenseLayer->pDst_->getData();
        fprintf(stderr, "%.1f\t", pDstData[i]);
    }
    fprintf(stderr, "\n\n");

    // int tt = 4*4 + 7*8 + 10*12 + 2*16 + 5*20 + 8*24 + 11*28;
    // fprintf(stderr, "%d\n", tt);

    return 0;
}
