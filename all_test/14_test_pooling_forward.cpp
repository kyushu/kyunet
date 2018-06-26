#include <time.h>       /* time */

#include "net.h"
#include "test_utils.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    // KyuNet Configuration
    int batchSize = 2;
    int height = 4;
    int width = 4;
    int channel = 1;



    // Config KyuNet
    KyuNet<float> net;
    InputLayer<float>* pInputLayer = (InputLayer<float> *)net.addInputLayer("input", batchSize, height, width, channel);

    ConvParam convParam{PaddingType::VALID, 1, 1, 0, 0/*, int dilation_w=1, int dilation_h=1*/};

    Layer<float> *pPoolingLayer = net.addPoolingLayer( pInputLayer, "Pooling", 2, 2, convParam, PoolingMethodType::MAX);

    // initialize KyuNet (allocate memory space)
    net.Compile(NetMode::TRAINING);

    // Set pseudo data
    // std::srand (time(NULL));
    Tensor<float>* pInputTensor = pInputLayer->pDst_;
    float* pInDstData = pInputTensor->getCPUData();
    for (int b = 0; b < batchSize; ++b)
    {
        for (int i = 0; i < pInputTensor->getWholeSize(); ++i)
        {
            float fval = (float(std::rand() % 100) / 100 - 0.5);
            pInDstData[i] = fval;
        }
    }

    // input
    int ih = pInputTensor->getHeight();
    int iw = pInputTensor->getWidth();
    int ic = pInputTensor->getChannel();
    fprintf(stderr, "### input data ###\n");
    print_matrix(batchSize, ic, ih, iw, pInDstData);

    //
    net.Forward();


    // output
    Tensor<float> *pDst = pPoolingLayer->pDst_;
    float* pDstData = pDst->getCPUData();
    // int batchsize = pDst->getNumOfData();
    int wholeSize = pDst->getWholeSize();
    int oh = pDst->getHeight();
    int ow = pDst->getWidth();
    int oc = pDst->getChannel();
    print_matrix(batchSize, oc, oh, ow, pDstData);

    // set pgDst
    float* pgDstData = pPoolingLayer->pgDst_->getCPUData();
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
