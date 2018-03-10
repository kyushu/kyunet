
#include "net.h"

void print_matrix(int batchSize, int channel, int height, int width, float* pData) {
    int size2D = height*width;
    int size3D = height*width*channel;
    for (int b = 0; b < batchSize; ++b)
    {
        fprintf(stderr, "batch: %d\n", b);
        for (int c = 0; c < channel; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    fprintf(stderr, "%f\t", pData[w + h*width + c*size2D + b*size3D]);
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
    int batchSize = 2;
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
                        0.5, 0.2, 0.5, 0.1,
                      0.8, 0.01, 0.01, 0.04,
                      0.01, 0.02, 0.02, 0.03,
                      0.03, 0.01, 0.01, 0.04};


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
    fprintf(stderr, "fc_unit(num of class) = %d\n", fc_unit);
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
    float* pInData = pInputLayer->pDst_->cpu_data();
    for (int i = 0; i < batchSize * height * width * channel; ++i)
    {
        pInData[i] = testData[i];
    }

    /*********************************************
     * Set initial value of weight of Dense layer
     *********************************************/
    int weight_wholeSize = pDenseLayer->pW_->WholeSize();
    float* pWData = pDenseLayer->pW_->cpu_data();
    for (int i = 0; i < weight_wholeSize; ++i)
    {
       pWData[i] = static_cast<float>(std::rand() % 100) / 100 - 0.5;
    }

    /*******************
     * Check Input Layer
     *******************/
    // Display data
    fprintf(stderr, "Input Data\n");
    int inWholeSize = pInputLayer->pDst_->WholeSize();
    print_matrix(batchSize, channel, height, width, pInData);

    /*******************
     * Check Weight of Dense Layer
     *******************/
    // Display weight
    fprintf(stderr, "weight of Dense Layer\n");
    int fc = pDenseLayer->pW_->Channel();
    int fh = pDenseLayer->pW_->Height();
    int fw = pDenseLayer->pW_->Width();
    print_matrix(1, fc, fh, fw, pDenseLayer->pW_->cpu_data());

    /***************
     * Test Forward
     ***************/
    net.Forward();

    /********************************
     * Check Dst data of Dense Layer
     ********************************/
    fprintf(stderr, "Dst Data (logits) of Dense Layer\n");
    int fc_dst_c = pDenseLayer->pDst_->Channel();
    int fc_dst_h = pDenseLayer->pDst_->Height();
    int fc_dst_w = pDenseLayer->pDst_->Width();
    int fc_dst_size2D = pDenseLayer->pDst_->Size2D();
    int fc_dst_size3D = pDenseLayer->pDst_->Size3D();
    print_matrix(batchSize, fc_dst_c, fc_dst_h, fc_dst_w, pDenseLayer->pDst_->cpu_data());

    /**********************************
     * Check Dst data of softmax layer
     **********************************/
    fprintf(stderr, "Dst data (probability) of Softmax Layer\n");
    int s_dst_c = pSoftmaxLayer->pDst_->Channel();
    int s_dst_h = pSoftmaxLayer->pDst_->Height();
    int s_dst_w = pSoftmaxLayer->pDst_->Width();
    print_matrix(2, s_dst_c, s_dst_h, s_dst_w, pSoftmaxLayer->pDst_->cpu_data());

    float sum = 0;
    float* softmax_dst_data = pSoftmaxLayer->pDst_->cpu_data();
    int sm_size3D = pSoftmaxLayer->pDst_->Size3D();
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
