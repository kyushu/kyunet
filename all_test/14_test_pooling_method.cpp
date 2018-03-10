#include <time.h>       /* time */

#include "net.h"
// #include "layer/pooling_layer.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    // Net Configuration
    int batchSize = 1;
    int height = 4;
    int width = 4;
    int channel = 1;

    Net net;

    // Config Net
    InputLayer* pInputLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);
    Layer *pPoolingLayer = net.addPoolingLayer( pInputLayer, "Pooling", 2, 2, 1, 1, 0, 0, PoolingMethodType::AVG);

    // initialize Net (allocate memory space)
    net.initialize();

    // Set pseudo data
    std::srand (time(NULL));
    Tensor* pInputTensor = pInputLayer->pDst_;
    float* pInDstData = pInputTensor->cpu_data();
    for (int i = 0; i < pInputTensor->WholeSize(); ++i)
    {
        float fval = (float(std::rand() % 100) / 100 - 0.5);
        pInDstData[i] = fval;
    }

    // input
    int ih = pInputTensor->Height();
    int iw = pInputTensor->Width();
    int ic = pInputTensor->Channel();
    for (int b = 0; b < batchSize; ++b)
    {
        for (int c = 0; c < ic; ++c)
        {
            for (int h = 0; h < ih; ++h)
            {
                for (int w = 0; w < iw; ++w)
                {
                    fprintf(stderr, "[%d] = %.3f\t", w + h*iw + c*ih*iw + b*ic*ih*iw, pInDstData[w + h*iw + c*ih*iw + b*ic*ih*iw]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }

    pPoolingLayer->Forward();


    // output
    Tensor *pDst = pPoolingLayer->pDst_;
    float* pDstData = pDst->cpu_data();
    int batchsize = pDst->NumOfData();
    int wholeSize = pDst->WholeSize();
    int oh = pDst->Height();
    int ow = pDst->Width();
    int oc = pDst->Channel();
    for (int b = 0; b < batchsize; ++b)
    {
        for (int c = 0; c < oc; ++c)
        {
            for (int h = 0; h < oh; ++h)
            {
                for (int w = 0; w < ow; ++w)
                {
                    fprintf(stderr, "[%d]=%.3f\t", w + h*ow + c*oh*ow + b*oc*oh*ow, pDstData[w + h*ow + c*oh*ow + b*oc*oh*ow]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
    }

    return 0;
}
