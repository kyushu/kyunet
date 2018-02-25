
#include "net.h"

void print_matrix(int batchSize, int channel, int height, int width, float* pData) {
    int size2D = height*width;
    int size3D = height*width*channel;
    for (int i = 0; i < batchSize; ++i)
    {
        for (int c = 0; c < channel; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    fprintf(stderr, "%.2f\t", pData[w + h*width + c * size2D + i * i*size3D]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "\n");
}

int main(int argc, char const *argv[])
{
    using namespace mkt;

    // Net Configuration
    int batchSize = 1;
    int height = 3;
    int width = 4;
    int channel = 1;

    fprintf(stderr, "[Input Data Info]\n");
    fprintf(stderr, "batchSize: %d\n", batchSize);
    fprintf(stderr, "channel  : %d\n", channel);
    fprintf(stderr, "height   : %d\n", height);
    fprintf(stderr, "width    : %d\n", width);


    /**************
     * pseudo data
     **************/
    float testData[] = {0.3, 0.4, 0.5, 0.6,
                        0.7, 0.4, 0.9, 0.1,
                        0.5, 0.2, 0.5, 0.1};


    /*************
     * Config Net
     *************/
    Net net;
    // Input Layer
    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);

    // Dense Layer
    InitializerType weightInitType = InitializerType::ONE;
    InitializerType biasInitType = InitializerType::ZERO;
    int fc_unit = 16;
    DenseLayer* pDenseLayer = (DenseLayer *)net.addDenseLayer(pInputLayer, "fc1", fc_unit, ActivationType::Relu, weightInitType, biasInitType);

    // Softmax Layer
    Layer* pSoftmaxLayer = net.addSoftmaxLayer( pDenseLayer, "softmax1");


    /*****************************************
     * Initialize Net (allocate memory space)
     *****************************************/
    net.initialize();

    /********************************
     * Fed Pseudo data to input layer
     ********************************/
    float* pInData = pInputLayer->pDst_->getData();
    for (int i = 0; i < batchSize * height * width * channel; ++i)
    {
        pInData[i] = testData[i];
    }

    /*********************************************
     * Set initial value of weight of Dense layer
     *********************************************/
    int weight_wholeSize = pDenseLayer->pW_->getWholeSize();
    float* pWData = pDenseLayer->pW_->getData();
    for (int i = 0; i < weight_wholeSize; ++i)
    {
       pWData[i] = static_cast<float>(std::rand() % 100) / 100 - 0.5;
    }

    /*******************
     * Check Input Layer
     *******************/
    // Display data
    fprintf(stderr, "Input Data\n");
    int inWholeSize = pInputLayer->pDst_->getWholeSize();
    print_matrix(batchSize, channel, height, width, pInData);
    // for (int i = 0; i < batchSize; ++i)
    // {
    //     for (int c = 0; c < channel; ++c)
    //     {
    //         fprintf(stderr, "ch: %d\n", c);
    //         for (int h = 0; h < height; ++h)
    //         {
    //             for (int w = 0; w < width; ++w)
    //             {
    //                 fprintf(stderr, "%.2f\t", pInData[w + h*width + c * height*width + i * i*inSize3D]);
    //             }
    //             fprintf(stderr, "\n");
    //         }
    //         fprintf(stderr, "\n");
    //     }
    // }

    /*******************
     * Check Weight of Dense Layer
     *******************/
    // Display weight
    fprintf(stderr, "weight of Dense Layer\n");
    int fc = pDenseLayer->pW_->getDepth();
    int fh = pDenseLayer->pW_->getHeight();
    int fw = pDenseLayer->pW_->getWidth();
    // float* pWData = pDenseLayer->pW_->getData();
    print_matrix(1, fc, fh, fw, pDenseLayer->pW_->getData());

    /***************
     * Test Forward
     ***************/
    net.forward();

    /********************************
     * Check Dst data of Dense Layer
     ********************************/
    fprintf(stderr, "Dst Data of Dense Layer\n");
    int fc_dst_c = pDenseLayer->pDst_->getDepth();
    int fc_dst_h = pDenseLayer->pDst_->getHeight();
    int fc_dst_w = pDenseLayer->pDst_->getWidth();
    int fc_dst_size2D = pDenseLayer->pDst_->getSize2D();
    int fc_dst_size3D = pDenseLayer->pDst_->getSize3D();
    // float* pFC_DstData = pDenseLayer->pDst_->getData();
    print_matrix(batchSize, fc_dst_c, fc_dst_h, fc_dst_w, pDenseLayer->pDst_->getData());

    /**********************************
     * Check Dst data of softmax layer
     **********************************/
    fprintf(stderr, "Dst data of Softmax Layer\n");
    int s_dst_c = pSoftmaxLayer->pDst_->getDepth();
    int s_dst_h = pSoftmaxLayer->pDst_->getHeight();
    int s_dst_w = pSoftmaxLayer->pDst_->getWidth();
    print_matrix(1, s_dst_c, s_dst_h, s_dst_w, pSoftmaxLayer->pDst_->getData());

    float sum = 0;
    float* softmax_dst_data = pSoftmaxLayer->pDst_->getData();
    for (int i = 0; i < pSoftmaxLayer->pDst_->getSize3D(); ++i)
    {
        sum += softmax_dst_data[i];
    }
    printf("sum all output of softmax: %f\n", sum);
    return 0;
}
