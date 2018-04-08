
#include "net.h"
#include "test_utils.hpp"

int main(int argc, char const *argv[])
{

    using namespace mkt;

    /* Parameters */
    int batchSize = 1;
    int input_height = 7;
    int input_width = 7;
    int input_ch = 1;

    /* Configure Net */
    Net net;

    // Input layer
    InputLayer* pInLayer = (InputLayer *)net.addInputLayer("input", batchSize, input_height, input_width, input_ch);

    // Convolution Layer 1
    LayerParams parConv1;
    parConv1.fc = 2;
    parConv1.fh = 3;
    parConv1.fw = 3;
    parConv1.stride_h = 1;
    parConv1.stride_w = 1;
    parConv1.pad_h = 0;
    parConv1.pad_w = 0;
    parConv1.padding_type = PaddingType::VALID;
    parConv1.actType = ActivationType::NONE;
    parConv1.weight_init_type = InitializerType::ONE;
    parConv1.bias_init_type = InitializerType::ZERO;
    ConvLayer* pConvLayer1 = (ConvLayer* )net.addConvLayer(pInLayer, "conv1", parConv1);

    // Convolution Layer 2
    LayerParams parConv2;
    parConv2.fc = 1;
    parConv2.fh = 3;
    parConv2.fw = 3;
    parConv2.stride_h = 1;
    parConv2.stride_w = 1;
    parConv2.pad_h = 0;
    parConv2.pad_w = 0;
    parConv2.padding_type = PaddingType::VALID;
    parConv2.actType = ActivationType::NONE;
    parConv2.weight_init_type = InitializerType::TEST;
    parConv2.bias_init_type = InitializerType::ZERO;
    ConvLayer* pConvLayer2 = (ConvLayer* )net.addConvLayer(pConvLayer1, "conv2", parConv2);

    // Pooling Layer
    LayerParams pool_params;
    pool_params.fh = 2;
    pool_params.fw = 2;
    pool_params.stride_h = 1;
    pool_params.stride_w = 1;
    pool_params.pad_h = 0;
    pool_params.pad_w = 0;
    pool_params.pooling_type = PoolingMethodType::AVG;
    PoolingLayer* pPoolingLayer = (PoolingLayer*)net.addPoolingLayer( pConvLayer2, "Pooling", pool_params);

    /* Initialize Net(Allocate memory) */
    net.Compile();


    /*Set random pesudo input data */
    float* pInData = pInLayer->pDst_->getCPUData();
    genRndPseudoData(pInData, batchSize, input_ch, input_height, input_width);

    /* Forward pass*/
    net.Forward();

    /* Display input data */
    fprintf(stderr, "############ [Input Data] ##########\n");
    print_matrix(batchSize, input_ch, input_height, input_width, pInData);

    /* Display relative value in matrix form */
    // Convolution Layer 1
    fprintf(stderr, "############ [Conv1-Weight] ############\n");
    float* pConv_WData1 = pConvLayer1->pW_->getCPUData();
    int fh1 = pConvLayer1->pW_->getHeight();
    int fw1 = pConvLayer1->pW_->getWidth();
    int fc1 = pConvLayer1->pW_->getChannel();
    print_matrix(1, fc1, fh1, fw1, pConv_WData1);

    fprintf(stderr, "############ [Conv1-TmpCol] ############\n");
    float* pConv_TmpColData1 = pConvLayer1->pTmpCol_->getCPUData();
    int tmpcol_c1 = pConvLayer1->pTmpCol_->getChannel();
    int tmpcol_h1 = pConvLayer1->pTmpCol_->getHeight();
    int tmpcol_w1 = pConvLayer1->pTmpCol_->getWidth();
    print_matrix(1, tmpcol_c1, tmpcol_h1, tmpcol_w1, pConv_TmpColData1);

    fprintf(stderr, "############ [Conv1-Output] ############\n");
    float* pConv_DstData1 = pConvLayer1->pDst_->getCPUData();
    int conv_wholeSize1 = pConvLayer1->pDst_->getWholeSize();
    int conv_oh1 = pConvLayer1->pDst_->getHeight();
    int conv_ow1 = pConvLayer1->pDst_->getWidth();
    int conv_oc1 = pConvLayer1->pDst_->getChannel();
    print_matrix(batchSize, conv_oc1, conv_oh1, conv_ow1, pConv_DstData1);

    // Convolution Layer 2
    fprintf(stderr, "############ [Conv2-Weight] ############\n");
    float* pConv_WData2 = pConvLayer2->pW_->getCPUData();
    int fh2 = pConvLayer2->pW_->getHeight();
    int fw2 = pConvLayer2->pW_->getWidth();
    int fc2 = pConvLayer2->pW_->getChannel();
    print_matrix(1, fc2, fh2, fw2, pConv_WData2);

    fprintf(stderr, "############ [Conv2-TmpCol] ############\n");
    float* pConv_TmpColData2 = pConvLayer2->pTmpCol_->getCPUData();
    int tmpcol_c2 = pConvLayer2->pTmpCol_->getChannel();
    int tmpcol_h2 = pConvLayer2->pTmpCol_->getHeight();
    int tmpcol_w2 = pConvLayer2->pTmpCol_->getWidth();
    print_matrix(1, tmpcol_c2, tmpcol_h2, tmpcol_w2, pConv_TmpColData2);

    fprintf(stderr, "############ [Conv2-Output] ############\n");
    float* pConv_DstData2 = pConvLayer2->pDst_->getCPUData();
    int conv_wholeSize2 = pConvLayer2->pDst_->getWholeSize();
    int conv_oh2 = pConvLayer2->pDst_->getHeight();
    int conv_ow2 = pConvLayer2->pDst_->getWidth();
    int conv_oc2 = pConvLayer2->pDst_->getChannel();
    print_matrix(batchSize, conv_oc2, conv_oh2, conv_ow2, pConv_DstData2);

    fprintf(stderr, "############ [Pooling Output] ############\n");
    float* pPooling_DstData = pPoolingLayer->pDst_->getCPUData();
    int pool_wholeSize = pPoolingLayer->pDst_->getWholeSize();
    int pool_oh = pPoolingLayer->pDst_->getHeight();
    int pool_ow = pPoolingLayer->pDst_->getWidth();
    int pool_oc = pPoolingLayer->pDst_->getChannel();
    print_matrix(batchSize, pool_oc, pool_oh, pool_ow, pPooling_DstData);

    /* Set pseudo pgDst value of Pooling layer */
    float* pgDstData = pPoolingLayer->pgDst_->getCPUData();
    for (int i = 0; i < pPoolingLayer->pgDst_->getWholeSize(); ++i)
    {
        float fval = float(std::rand() % 10);
            pgDstData[i] = fval;
    }
    /* Display pseudo pgDst in matrix form */
    fprintf(stderr, "############ [grad output of pooling] ############\n");
    print_matrix(batchSize, pool_oc, pool_oh, pool_ow, pgDstData);

    net.Backward();

    fprintf(stderr, "############ [Conv2-Col2Im] ############\n");
    float* pConv_col2im2 = pConvLayer2->pTmpCol_->getCPUData();
    int c2_col2im_c = pConvLayer2->pTmpCol_->getChannel();
    int c2_col2im_h = pConvLayer2->pTmpCol_->getHeight();
    int c2_col2im_w = pConvLayer2->pTmpCol_->getWidth();

    print_matrix(1, c2_col2im_c, c2_col2im_h, c2_col2im_w, pConv_col2im2);


    fprintf(stderr, "############ [Conv2-Gradient] ############\n");
    float* pConv_gDstData2 = pConvLayer2->pgDst_->getCPUData();
    print_matrix(batchSize, conv_oc2, conv_oh2, conv_ow2, pConv_gDstData2);

    fprintf(stderr, "############ [Conv1-Gradient] ############\n");
    float* pConv_gDstData1 = pConvLayer1->pgDst_->getCPUData();
    print_matrix(batchSize, conv_oc1, conv_oh1, conv_ow1, pConv_gDstData1);

    return 0;
}
