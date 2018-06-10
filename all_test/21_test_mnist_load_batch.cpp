#include <vector>
#include <fstream>
#include <string>     // std::string, std::stoi

#include "stb_image.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"
#include "solver/sgd_solver.h"
#include "utils.hpp"

#include "test_utils.hpp"


using namespace mkt;

void display_image_with_label(std::vector<std::string> image_files, std::vector<int> labels, std::string root="", int number=10) {

    for (int i = 0; i < number; ++i)
    {
        std::string file = image_files[i];
        std::string full_path = root + file;
        cv::Mat image = cv::imread(full_path);
        cv::Mat enlarge_image;
        cv::resize(image, enlarge_image, cv::Size(150,150));

        std::string label = std::to_string(labels[i]);

        cv::putText(enlarge_image, label, cv::Point(10,40), 0, 1, cv::Scalar(255,255,255),3);

        cv::imshow("image", enlarge_image);
        char key = (char)cv::waitKey();
        if (key == 27)
        {
            break;
        }
    }
}

void check_image_info(std::string full_path) {
    int w, h, c;
    unsigned char *pImg = stbi_load(full_path.c_str(), &w, &h, &c, 0);
    fprintf(stderr, "w: %d, h: %d, c: %d\n", w, h, c);
}

template<typename T>
KyuNet<T>* configure_kyunet(int batchSize, int input_height, int input_width, int input_ch) {



    /* Configure Net */
    KyuNet<T>* net = new KyuNet<T>{};

    // Input layer
    InputLayer<T>* pInLayer = (InputLayer<T> *)net->addInputLayer("input", batchSize, input_height, input_width, input_ch);

    fprintf(stderr, "input Layer: batch: %d, h: %d, w: %d, c: %d (%d)\n",
        pInLayer->pDst_->getNumOfData(),
        pInLayer->pDst_->getHeight(),
        pInLayer->pDst_->getWidth(),
        pInLayer->pDst_->getChannel(),
        pInLayer->pDst_->getWholeSize());

    // Convolution Layer 1
    LayerParams conv1_par;
    conv1_par.fc = 8;
    conv1_par.fh = 3;
    conv1_par.fw = 3;
    conv1_par.stride_h = 1;
    conv1_par.stride_w = 1;
    conv1_par.pad_h = 0;
    conv1_par.pad_w = 0;
    conv1_par.padding_type = PaddingType::VALID;
    conv1_par.actType = ActivationType::RELU;
    conv1_par.weight_init_type = InitializerType::HE_INIT_NORM;
    conv1_par.bias_init_type = InitializerType::ZERO;
    ConvLayer<T>* pConvLayer1 = (ConvLayer<T>* )net->addConvLayer(pInLayer, "conv1", conv1_par);

    // Pooling Layer
    LayerParams pool_params;
    pool_params.fh = 2;
    pool_params.fw = 2;
    pool_params.stride_h = 1;
    pool_params.stride_w = 1;
    pool_params.pad_h = 0;
    pool_params.pad_w = 0;
    pool_params.pooling_type = PoolingMethodType::MAX;
    PoolingLayer<T>* pPoolingLayer = (PoolingLayer<T>*)net->addPoolingLayer( pConvLayer1, "Pooling1", pool_params);

    // Dense Layer
    LayerParams dense_params;
    dense_params.fc = 10;
    dense_params.actType = ActivationType::RELU;
    dense_params.weight_init_type = InitializerType::HE_INIT_NORM;
    dense_params.bias_init_type = InitializerType::ZERO;
    DenseLayer<T>* pDenseLayer = (DenseLayer<T>* )net->addDenseLayer(pPoolingLayer, "dense1", dense_params);

    // CrossEntropyWutgSoftmaxLayer
    CrossEntropyLossWithSoftmaxLayer<T>* pCrossEntropyLayer = (CrossEntropyLossWithSoftmaxLayer<T> *)net->addCrossEntropyLossWithSoftmaxLayer(pDenseLayer, "cross_entropy_loss");

    /* Initialize Net (Allocate menory) */
    net->Compile(NetMode::TRAINING);

    SGDSolver<T>* pSgdSolver = new SGDSolver<T>{net};
    pSgdSolver->initialize();
    net->addSolver(pSgdSolver);

    return net;
}

int main(int argc, char const *argv[])
{
    /**
     * the format of mnist_train_file is image_file_path label
     * For example:
     * /mtdata/Download/mnist/images/86.png 8
     */
    // std::string root = "../example_data/mnist/cat_train_images/";
    // std::string mnist_train_file = "../example_data/mnist/cat_train_images/train_0_1.list";
    std::string root = "../example_data/mnist/";
    std::string mnist_train_file = "../example_data/mnist/training_file.txt";


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

    // display_image_with_label(image_files, labels, root, 20);
    check_image_info(image_files[0]);


    /* Parameters */
    int batchSize = 64;
    int input_height = 28;
    int input_width = 28;
    int input_ch = 1;

    // KyuNet* net = configure_kyunet(batchSize, input_height, input_width, input_ch);

    //////////////////////////////////////////////
    /* Configure Net */
    KyuNet<float>* net = new KyuNet<float>{};

    // Input layer
    InputLayer<float>* pInLayer = (InputLayer<float> *)net->addInputLayer("input", batchSize, input_height, input_width, input_ch);

    fprintf(stderr, "input Layer: batch: %d, h: %d, w: %d, c: %d (%d)\n",
        pInLayer->pDst_->getNumOfData(),
        pInLayer->pDst_->getHeight(),
        pInLayer->pDst_->getWidth(),
        pInLayer->pDst_->getChannel(),
        pInLayer->pDst_->getWholeSize());

    // Convolution Layer 1
    LayerParams conv1_par;
    conv1_par.fc = 32;
    conv1_par.fh = 3;
    conv1_par.fw = 3;
    conv1_par.stride_h = 1;
    conv1_par.stride_w = 1;
    conv1_par.pad_h = 0;
    conv1_par.pad_w = 0;
    conv1_par.padding_type = PaddingType::VALID;
    conv1_par.actType = ActivationType::RELU;
    conv1_par.weight_init_type = InitializerType::HE_INIT_NORM;
    conv1_par.bias_init_type = InitializerType::ZERO;
    ConvLayer<float>* pConvLayer1 = (ConvLayer<float>* )net->addConvLayer(pInLayer, "conv1", conv1_par);

    // Convolution Layer 2
    LayerParams conv2_par;
    conv2_par.fc = 64;
    conv2_par.fh = 3;
    conv2_par.fw = 3;
    conv2_par.stride_h = 1;
    conv2_par.stride_w = 1;
    conv2_par.pad_h = 0;
    conv2_par.pad_w = 0;
    conv2_par.padding_type = PaddingType::VALID;
    conv2_par.actType = ActivationType::RELU;
    conv2_par.weight_init_type = InitializerType::HE_INIT_NORM;
    conv2_par.bias_init_type = InitializerType::ZERO;
    ConvLayer<float>* pConvLayer2 = (ConvLayer<float>* )net->addConvLayer(pConvLayer1, "conv2", conv2_par);

    // Pooling Layer
    LayerParams pool_params;
    pool_params.fh = 2;
    pool_params.fw = 2;
    pool_params.stride_h = 1;
    pool_params.stride_w = 1;
    pool_params.pad_h = 0;
    pool_params.pad_w = 0;
    pool_params.pooling_type = PoolingMethodType::MAX;
    PoolingLayer<float>* pPoolingLayer = (PoolingLayer<float>*)net->addPoolingLayer( pConvLayer2, "Pooling1", pool_params);

    // Dense Layer
    LayerParams dense_params;
    dense_params.fc = 10;
    dense_params.actType = ActivationType::RELU;
    dense_params.weight_init_type = InitializerType::HE_INIT_NORM;
    dense_params.bias_init_type = InitializerType::ZERO;
    DenseLayer<float>* pDenseLayer = (DenseLayer<float>* )net->addDenseLayer(pPoolingLayer, "dense1", dense_params);

    // CrossEntropyWutgSoftmaxLayer
    CrossEntropyLossWithSoftmaxLayer<float>* pCrossEntropyLayer = (CrossEntropyLossWithSoftmaxLayer<float> *)net->addCrossEntropyLossWithSoftmaxLayer(pDenseLayer, "cross_entropy_loss");

    float learning_rate = 0.0003;
    SGDSolver<float>* pSgdSolver = new SGDSolver<float>{net, learning_rate};
    net->addSolver(pSgdSolver);

    /* Initialize Net (Allocate menory) */
    net->Compile(NetMode::TRAINING);

    //////////////////////////////////////////////



    int num_batch = static_cast<float>(image_files.size()) / batchSize;
    fprintf(stderr, "num_batch: %d\n", num_batch);
    for (int i = 0; i < 50; ++i)
    {
        for (int b = 0; b < num_batch; ++b)
        {
            // fprintf(stderr, "batch index: %d\n", b);

            int start = b * batchSize;
            int end   = (b+1) * batchSize;


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


            net->Train();
            if (b > 0 && b%10 == 0)
            {
                float loss = pCrossEntropyLayer->pDst_->getCPUData()[0];
                fprintf(stderr, "%d: loss = %f\n", i, loss);
            }
        }
    }


    return 0;
}
