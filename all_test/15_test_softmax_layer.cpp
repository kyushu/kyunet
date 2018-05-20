#include "net.h"
#include "test_utils.hpp"

int main(int argc, char const *argv[])
{
    using namespace mkt;

    // KyuNet Configuration
    int batchSize = 2;
    int height = 1;
    int width = 1;
    int channel = 12;

    fprintf(stderr, "[Input Data Info]\n");
    fprintf(stderr, "batchSize: %d\n", batchSize);
    fprintf(stderr, "channel  : %d\n", channel);
    fprintf(stderr, "height   : %d\n", height);
    fprintf(stderr, "width    : %d\n", width);


    /**************
     * pseudo data
     **************/
    float testData[] = {3, 4, 5, 6,
                        7, 4, 9, 1,
                        5, 2, 5, 1,
                      8,  1, 1, 4,
                      1, 2, 2, 3,
                      3, 1, 1, 4};


    /*************
     * Config KyuNet
     *************/
    KyuNet<float> net;
    // Input Layer
    InputLayer<float>* pInputLayer = (InputLayer<float> *)net.addInputLayer("input", batchSize, height, width, channel);

    // Softmax Layer
    Layer<float>* pSoftmaxLayer = net.addSoftmaxLayer( pInputLayer, "softmax1");


    /*****************************************
     * Initialize KyuNet (allocate memory space)
     *****************************************/
    net.Compile(NetMode::TRAINING);

    /********************************
     * Fed Pseudo data to input layer
     ********************************/
    float* pInData = pInputLayer->pDst_->getCPUData();
    for (int i = 0; i < batchSize * height * width * channel; ++i)
    {
        pInData[i] = testData[i];
    }

    /*******************
     * Check Input Layer
     *******************/
    // Display data
    fprintf(stderr, "Input Data\n");
    // int inWholeSize = pInputLayer->pDst_->getWholeSize();
    print_matrix(batchSize, channel, height, width, pInData);


    /***************
     * Test Forward
     ***************/
    net.Forward();

    /**********************************
     * Check Dst data of softmax layer
     **********************************/
    fprintf(stderr, "Dst data (probability) of Softmax Layer\n");
    int s_dst_c = pSoftmaxLayer->pDst_->getChannel();
    int s_dst_h = pSoftmaxLayer->pDst_->getHeight();
    int s_dst_w = pSoftmaxLayer->pDst_->getWidth();
    print_matrix(2, s_dst_c, s_dst_h, s_dst_w, pSoftmaxLayer->pDst_->getCPUData());

    float sum = 0;
    float* softmax_dst_data = pSoftmaxLayer->pDst_->getCPUData();
    int sm_size3D = pSoftmaxLayer->pDst_->getSize3D();
    for (int b = 0; b < batchSize; ++b)
    {
        fprintf(stderr, "batch: %d\n", b);
        sum = 0;
        for (int i = 0; i < sm_size3D; ++i)
        {
            fprintf(stderr, "sum = %f, val[%d]=%f\n", sum, i + b*sm_size3D, softmax_dst_data[i + b*sm_size3D]);
            sum += softmax_dst_data[i + b*sm_size3D];

        }
        printf("sum all output of softmax: %f\n", sum);
    }

    return 0;
}
