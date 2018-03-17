#include <time.h>       /* time */

#include "net.h"
#include "test_utils.hpp"

using namespace mkt;

int main(int argc, char const *argv[])
{

    // Net Configuration
    int batchSize = 2;
    int height = 4;
    int width = 4;
    int channel = 1;



    // Config Net
    Net net;
    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);
    Layer *pPoolingLayer = net.addPoolingLayer( pInputLayer, "Pooling", 2, 2, 1, 1, 0, 0, PoolingMethodType::MAX);

    // initialize Net (allocate memory space)
    net.initialize();

    // Set pseudo data
    // std::srand (time(NULL));
    Tensor* pInputTensor = pInputLayer->pDst_;
    float* pInDstData = pInputTensor->cpu_data();
    for (int b = 0; b < batchSize; ++b)
    {
        for (int i = 0; i < pInputTensor->WholeSize(); ++i)
        {
            float fval = (float(std::rand() % 100) / 100 - 0.5);
            pInDstData[i] = fval;
        }
    }

    // input
    int ih = pInputTensor->Height();
    int iw = pInputTensor->Width();
    int ic = pInputTensor->Channel();
    fprintf(stderr, "### input data ###\n");
    print_matrix(batchSize, ic, ih, iw, pInDstData);

    //
    net.Forward();


    // output
    Tensor *pDst = pPoolingLayer->pDst_;
    float* pDstData = pDst->cpu_data();
    int batchsize = pDst->NumOfData();
    int wholeSize = pDst->WholeSize();
    int oh = pDst->Height();
    int ow = pDst->Width();
    int oc = pDst->Channel();
    print_matrix(batchSize, oc, oh, ow, pDstData);

    // set pgDst
    float* pgDstData = pPoolingLayer->pgDst_->cpu_data();
    for (int i = 0; i < wholeSize; ++i)
    {
        float fval = (float(std::rand() % 100) / 100 - 0.5);
            pgDstData[i] = fval;
    }
    fprintf(stderr, "### grad output of pooling ###\n");
    print_matrix(batchSize, oc, oh, ow, pgDstData);


    net.Backward();



    return 0;
}
