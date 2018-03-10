
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

    int batchSize = 1;
    int height = 1;
    int width = 1;
    int channel = 12;

    // float testData[] = {3, 4, 5, 6,
    //                     7, 4, 9, 1,
    //                     5, 2, 5, 1,
    //                   8,  1, 1, 4,
    //                   1, 2, 2, 3,
    //                   3, 1, 1, 4};

    float testData[] = {8, 1, 1, 4,
                        1, 2, 2, 3,
                        3, 1, 1, 4};

    int label[] = {0};

     /*************
     * Config Net
     *************/
    Net net;

    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);

    CrossEntropyLossWithSoftmaxLayer* pCrossEntropyLayer = (CrossEntropyLossWithSoftmaxLayer *)net.addCrossEntropyLossWithSoftmaxLayer(pInputLayer, "cross_entropy_loss");


    net.initialize();

    /********************************
     * load Pseudo data to input layer
     ********************************/
    float* pInData = pInputLayer->pDst_->cpu_data();
    for (int i = 0; i < batchSize * height * width * channel; ++i)
    {
        pInData[i] = testData[i];
    }

    /***************************
     * load label to loss layer
     ***************************/
    pCrossEntropyLayer->LoadLabel(batchSize, label);

    net.Forward();

    /*
     * Display Probability
     */
    Layer* pSoftmaxLayer = &(pCrossEntropyLayer->softmaxLayer_);
    fprintf(stderr, "Dst data (probability) of Softmax Layer\n");
    int s_dst_c = pSoftmaxLayer->pDst_->Channel();
    int s_dst_h = pSoftmaxLayer->pDst_->Height();
    int s_dst_w = pSoftmaxLayer->pDst_->Width();
    //

    float sum = 0;
    float* softmax_dst_data = pSoftmaxLayer->pDst_->cpu_data();
    print_matrix(batchSize, s_dst_c, s_dst_h, s_dst_w, softmax_dst_data);
    int sm_size3D = pSoftmaxLayer->pDst_->Size3D();
    fprintf(stderr, "[Verify] each probability of each batch \n");
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

    float* loss = pCrossEntropyLayer->pDst_->cpu_data();
    fprintf(stderr, "loss: %f\n", loss[0]);

    return 0;
}
