
#include <vector>
#include <fstream>
#include <string>     // std::string, std::stoi

#include "net.h"

#include "utils.hpp"

#include "test_utils.hpp"

int main(int argc, char const *argv[])
{

    using namespace mkt;


    /**
     * the format of mnist_train_file is image_file_path label
     * For example:
     * /mtdata/Download/mnist/images/86.png 8
     */
    std::string root = "../example_data/mnist/cat_train_images/";
    std::string mnist_train_file = "../example_data/mnist/cat_train_images/train_0_1.list";

    std::ifstream ifile(mnist_train_file);
    if (!ifile.good())
    {
        fprintf(stderr, "can't open %s\n", mnist_train_file.c_str());
        return -1;
    }

    std::vector<std::string> image_files;
    std::vector<int> labels;
    std::string line;
    while(std::getline(ifile, line)) {
        // fprintf(stderr, "%s\n", line.c_str());
        std::vector<std::string> compons = UTILS::split(line, ' ');
        image_files.push_back(root + compons[0]);
        // int label = std::stoi(compons[1]);
        labels.emplace_back(std::stoi(compons[1]));
    }

    /* Parameters */
    int batchSize = 3;
    int input_height = 28;
    int input_width = 28;
    int input_ch = 1;

    /* Configure Net */
    KyuNet* net = new KyuNet{};

    // Input layer
    InputLayer* pInLayer = (InputLayer *)net->addInputLayer("input", batchSize, input_height, input_width, input_ch);

    // Convolution Layer 1
    LayerParams conv1_par;
    conv1_par.fc = 2;
    conv1_par.fh = 7;
    conv1_par.fw = 7;
    conv1_par.stride_h = 3;
    conv1_par.stride_w = 3;
    conv1_par.pad_h = 0;
    conv1_par.pad_w = 0;
    conv1_par.padding_type = PaddingType::VALID;
    conv1_par.actType = ActivationType::RELU;
    conv1_par.weight_init_type = InitializerType::HE_INIT_NORM;
    conv1_par.bias_init_type = InitializerType::ZERO;
    ConvLayer* pConvLayer1 = (ConvLayer* )net->addConvLayer(pInLayer, "conv1", conv1_par);

    // Pooling Layer
    LayerParams pool_params;
    pool_params.fh = 2;
    pool_params.fw = 2;
    pool_params.stride_h = 1;
    pool_params.stride_w = 1;
    pool_params.pad_h = 0;
    pool_params.pad_w = 0;
    pool_params.pooling_type = PoolingMethodType::MAX;
    PoolingLayer* pPoolingLayer = (PoolingLayer*)net->addPoolingLayer( pConvLayer1, "Pooling1", pool_params);

    // Dense Layer
    LayerParams dense_params;
    dense_params.fc = 2;
    dense_params.actType = ActivationType::RELU;
    dense_params.weight_init_type = InitializerType::HE_INIT_NORM;
    dense_params.bias_init_type = InitializerType::ZERO;
    DenseLayer* pDenseLayer = (DenseLayer* )net->addDenseLayer(pPoolingLayer, "dense1", dense_params);

    // CrossEntropyWutgSoftmaxLayer
    CrossEntropyLossWithSoftmaxLayer* pCrossEntropyLayer = (CrossEntropyLossWithSoftmaxLayer *)net->addCrossEntropyLossWithSoftmaxLayer(pDenseLayer, "cross_entropy_loss");

    fprintf(stderr, "num of layer: %d\n", net->getNumOfLayer());

    /* Initialize Net (Allocate menory) */
    net->Compile();

    // /*Set random pesudo input data */
    float* pInData = pInLayer->pDst_->getCPUData();
    // genRndPseudoData(pInData, batchSize, input_ch, input_height, input_width);

    // /* load label to loss layer */
    // int label[] = {0, 1, 2};
    // pCrossEntropyLayer->LoadLabel(batchSize, label);



    /* Forward pass */
    int num_batch = static_cast<float>(image_files.size()) / batchSize;
    fprintf(stderr, "num_batch: %d\n", num_batch);
    for (int i = 0; i < 4; ++i)
    {
        fprintf(stderr, "batch index: %d\n", i);

        int start = i * batchSize;
        int end = (i+1) * batchSize;

        // batch for image file
        std::vector<std::string>::const_iterator img_first = image_files.begin() + start;
        std::vector<std::string>::const_iterator img_last = image_files.begin() + end;
        std::vector<std::string> batchFile(img_first, img_last);
        net->add_data_from_file_list(batchFile);
        // Batch for label
        std::vector<int>::const_iterator label_first = labels.begin() + start;
        std::vector<int>::const_iterator label_last = labels.begin() + end;
        std::vector<int> batchLabel(label_first, label_last);
        net->addBatchLabels("cross_entropy_loss", batchLabel);

        net->Forward();

        /* Display input data */
        // fprintf(stderr, "############ [Input Data] ##########\n");
        // print_matrix(batchSize, input_ch, input_height, input_width, pInData);

        /* Display relative value in matrix form */
        // Convolution Layer 1
        fprintf(stderr, "############ [Conv1-Weight] ############\n");
        float* pConv_WData1 = pConvLayer1->pW_->getCPUData();
        int fh1 = pConvLayer1->pW_->getHeight();
        int fw1 = pConvLayer1->pW_->getWidth();
        int fc1 = pConvLayer1->pW_->getChannel();
        print_matrix(1, fc1, fh1, fw1, pConv_WData1);

        // fprintf(stderr, "############ [Conv1-TmpCol(Im2Col)] ############\n");
        // float* pConv_TmpColData1 = pConvLayer1->pTmpCol_->getCPUData();
        // int tmpcol_c1 = pConvLayer1->pTmpCol_->getChannel();
        // int tmpcol_h1 = pConvLayer1->pTmpCol_->getHeight();
        // int tmpcol_w1 = pConvLayer1->pTmpCol_->getWidth();
        // print_matrix(1, tmpcol_c1, tmpcol_h1, tmpcol_w1, pConv_TmpColData1);

        fprintf(stderr, "############ [Conv1-Output] ############\n");
        float* pConv_DstData1 = pConvLayer1->pDst_->getCPUData();
        int conv_wholeSize1 = pConvLayer1->pDst_->getWholeSize();
        int conv_oh1 = pConvLayer1->pDst_->getHeight();
        int conv_ow1 = pConvLayer1->pDst_->getWidth();
        int conv_oc1 = pConvLayer1->pDst_->getChannel();
        print_matrix(batchSize, conv_oc1, conv_oh1, conv_ow1, pConv_DstData1);

        fprintf(stderr, "############ [Pooling Output] ############\n");
        float* pPooling_DstData = pPoolingLayer->pDst_->getCPUData();
        int pool_wholeSize = pPoolingLayer->pDst_->getWholeSize();
        int pool_oh = pPoolingLayer->pDst_->getHeight();
        int pool_ow = pPoolingLayer->pDst_->getWidth();
        int pool_oc = pPoolingLayer->pDst_->getChannel();
        print_matrix(batchSize, pool_oc, pool_oh, pool_ow, pPooling_DstData);

        fprintf(stderr, "############ [Dense-Weight] ############\n");
        float* pDense_WData1 = pDenseLayer->pW_->getCPUData();
        int fc_fh1 = pDenseLayer->pW_->getHeight();
        int fc_fw1 = pDenseLayer->pW_->getWidth();
        int fc_fc1 = pDenseLayer->pW_->getChannel();
        print_matrix(1, fc_fh1, fc_fw1, fc_fc1, pDense_WData1);

        fprintf(stderr, "############ [Dense Output](logits) ############\n");
        int fc_dst_c = pDenseLayer->pDst_->getChannel();
        int fc_dst_h = pDenseLayer->pDst_->getHeight();
        int fc_dst_w = pDenseLayer->pDst_->getWidth();
        int fc_dst_size2D = pDenseLayer->pDst_->getSize2D();
        int fc_dst_size3D = pDenseLayer->pDst_->getSize3D();
        print_matrix(batchSize, fc_dst_c, fc_dst_h, fc_dst_w, pDenseLayer->pDst_->getCPUData());

        /*
         * Display Probability
         */
        fprintf(stderr, "############ [Softmax Output](Probability) ############\n");
        Layer* pSoftmaxLayer = &(pCrossEntropyLayer->softmaxLayer_);
        int s_dst_c = pSoftmaxLayer->pDst_->getChannel();
        int s_dst_h = pSoftmaxLayer->pDst_->getHeight();
        int s_dst_w = pSoftmaxLayer->pDst_->getWidth();

        float sum = 0;
        float* softmax_dst_data = pSoftmaxLayer->pDst_->getCPUData();
        print_matrix(batchSize, s_dst_c, s_dst_h, s_dst_w, softmax_dst_data);
        int sm_size3D = pSoftmaxLayer->pDst_->getSize3D();
        fprintf(stderr, "[Verify] each probability of each batch \n");
        for (int b = 0; b < batchSize; ++b)
        {
            fprintf(stderr, "[batch]: %d\n", b);
            sum = 0;
            for (int i = 0; i < sm_size3D; ++i)
            {
                fprintf(stderr, "sum = %f, val[%d]=%f\n", sum, i + b*sm_size3D, softmax_dst_data[i + b*sm_size3D]);
                sum += softmax_dst_data[i + b*sm_size3D];
            }
            printf("sum all output of softmax: %f\n", sum);
        }

        float* loss = pCrossEntropyLayer->pDst_->getCPUData();
        fprintf(stderr, "loss: %f\n", loss[0]);


        fprintf(stderr, "################################################\n");
        fprintf(stderr, "############ [Backpropagation] ############\n");

    }

    // net.Backward();


    return 0;
}
