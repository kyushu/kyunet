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

    // Preset test data
    unsigned char s0[] = {1,2, 4,5, 7,8, 10,11};
        /*13,14,15, 16,17,18, 19,20,21, 22,23,24, 25,26,27, 28,29,30, 31,32,33, 34,35,36};*/

    unsigned char s1[] = {1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3};
    unsigned char s2[] = {4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6};
    unsigned char s3[] = {7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9};

    int channel = 2/*3*/;
    int height = 2/*4*/;
    int width = 2/*3*/;
    int batchSize = 1;

    // Configure, initialize network
    Net net;
    net.addInputLayer("input", batchSize, height, width, channel);
    Layer* pDenseLayer = net.addDenseLayer("Layer1", 4);
    net.initialize();

    // Get InputLayer
    InputLayer* pInput = net.getInputLayer();
    const float *pdata = pInput->pDst->getData();
    int size3D = pInput->pDst->getSize3D();
    fprintf(stderr, "input size3D: %d\n", size3D);

    // Add data
    pInput->FlattenImageToTensor(s0, false);
    // pInput->FlattenImageToTensor(s1, false);
    // pInput->FlattenImageToTensor(s2, true);
    // pInput->FlattenImageToTensor(s3, true);

    int weightsize3D = pDenseLayer->pW->getSize3D()
    int weightWholeSize = pDenseLayer->pW->getWholeSize();
    fprintf(stderr, "dense weight size3D: %d\n", weightsize3D);
    fprintf(stderr, "dense weight WholeSize: %d\n", weightWholeSize);

    // for (int i = 0; i < weightWholeSize; ++i)
    // {
    //     fprintf(stderr, "%d: %f\n", i, pDenseLayer->pW->pData[i]);
    // }

    pDenseLayer->forward();
    for (int i = 0; i < pDenseLayer->pDst->wholeSize; ++i)
    {
        fprintf(stderr, "%d- %f\n", i, pDenseLayer->pDst->pData[i]);
    }

    // int tt = 4*4 + 7*8 + 10*12 + 2*16 + 5*20 + 8*24 + 11*28;
    // fprintf(stderr, "%d\n", tt);

    return 0;
}
