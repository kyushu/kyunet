
#include "stdlib.h"
#include "net.h"
#include "test_utils.hpp"


int main(int argc, char const *argv[])
{

    using namespace mkt;

    /* Parameters */
    int batchSize = 1;
    int height = 6;
    int width = 6;
    int channel = 1;


    /*Configure Net*/
    Net net;
    // Input layer
    InputLayer* pInLayer = (InputLayer *)net.addInputLayer("input", batchSize, height, width, channel);
    // Conv layer
    LayerParams conv_params;
    conv_params.fc = 2;
    conv_params.fh = 3;
    conv_params.fw = 3;
    conv_params.stride_h = 1;
    conv_params.stride_w = 1;
    conv_params.pad_h = 0;
    conv_params.pad_w = 0;
    conv_params.padding_type = PaddingType::VALID;
    conv_params.actType = ActivationType::NONE;
    conv_params.weight_init_type = InitializerType::TEST;
    conv_params.bias_init_type = InitializerType::ZERO;
    ConvLayer* pConvLayer = (ConvLayer*)net.addConvLayer(pInLayer, "conv_1", conv_params);

    // InitializerType weightInitType = InitializerType::TEST;
    // InitializerType biasInitType = InitializerType::ZERO;
    // ConvLayer* pConvLayer = (ConvLayer*)net.addConvLayer(pInLayer, "conv_1", 3, 3, 2, 1, 1, 0, 0, PaddingType::VALID, ActivationType::NONE, weightInitType, biasInitType);

    // Pooling layer
    // PoolingLayer* pPoolingLayer = (PoolingLayer*)net.addPoolingLayer( pConvLayer, "Pooling", 2, 2, 1, 1, 0, 0, PoolingMethodType::MAX);

    LayerParams pool_params;
    pool_params.fh = 2;
    pool_params.fw = 2;
    pool_params.stride_h = 1;
    pool_params.stride_w = 1;
    pool_params.pad_h = 0;
    pool_params.pad_w = 0;
    pool_params.pooling_type = PoolingMethodType::AVG;
    PoolingLayer* pPoolingLayer = (PoolingLayer*)net.addPoolingLayer( pConvLayer, "Pooling", pool_params);

    /*Initialize Net*/
    net.initialize();

    /* Generate pseudo Data */
    float* pInData = pInLayer->pDst_->cpu_data();

    for (int b = 0; b < batchSize; ++b)
    {
        for (int c = 0; c < channel; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    float rndVal = (rand() % 1000) / 1000.0f;
                    pInData[w + h*width + c*(height*width) + b*(channel*height*width)] = rndVal;
                    // fprintf(stderr, "data[%d]=%f\n", w + h*width + c*(height*width) + b*(channel*height*width), rndVal);
                }
            }
        }
    }
    /* Display Input Data */
    fprintf(stderr, "############ [Input Data] ##########\n");
    print_matrix(batchSize, channel, height, width, pInData);

    net.Forward();


    fprintf(stderr, "############ [Conv-Weight] ############\n");
    float* pConv_WData = pConvLayer->pW_->cpu_data();
    int fh = pConvLayer->pW_->Height();
    int fw = pConvLayer->pW_->Width();
    int fc = pConvLayer->pW_->Channel();
    print_matrix(1, fc, fh, fw, pConv_WData);

    fprintf(stderr, "############ [Conv-TmpCol] ############\n");
    float* pConv_TmpColData = pConvLayer->pTmpCol_->cpu_data();
    int tmpcol_c = pConvLayer->pTmpCol_->Channel();
    int tmpcol_h = pConvLayer->pTmpCol_->Height();
    int tmpcol_w = pConvLayer->pTmpCol_->Width();
    print_matrix(1, tmpcol_c, tmpcol_h, tmpcol_w, pConv_TmpColData);

    fprintf(stderr, "############ [Conv-Output] ############\n");
    float* pConv_DstData = pConvLayer->pDst_->cpu_data();
    int conv_wholeSize = pConvLayer->pDst_->WholeSize();
    int conv_oh = pConvLayer->pDst_->Height();
    int conv_ow = pConvLayer->pDst_->Width();
    int conv_oc = pConvLayer->pDst_->Channel();
    print_matrix(batchSize, conv_oc, conv_oh, conv_ow, pConv_DstData);

    fprintf(stderr, "############ [Pooling Output] ############\n");
    float* pPooling_DstData = pPoolingLayer->pDst_->cpu_data();
    int pool_wholeSize = pPoolingLayer->pDst_->WholeSize();
    int pool_oh = pPoolingLayer->pDst_->Height();
    int pool_ow = pPoolingLayer->pDst_->Width();
    int pool_oc = pPoolingLayer->pDst_->Channel();
    print_matrix(batchSize, pool_oc, pool_oh, pool_ow, pPooling_DstData);



    // set pseudo pgDst
    float* pgDstData = pPoolingLayer->pgDst_->cpu_data();
    for (int i = 0; i < pPoolingLayer->pgDst_->WholeSize(); ++i)
    {
        float fval = (float(std::rand() % 100) / 100 - 0.5);
            pgDstData[i] = fval;
    }
    fprintf(stderr, "############ [grad output of pooling] ############\n");
    print_matrix(batchSize, pool_oc, pool_oh, pool_ow, pgDstData);


    pPoolingLayer->Backward();

    fprintf(stderr, "############ [Conv-Gradient] ############\n");
    float* pConv_gDstData = pConvLayer->pgDst_->cpu_data();
    print_matrix(batchSize, conv_oc, conv_oh, conv_ow, pConv_gDstData);

    return 0;
}
