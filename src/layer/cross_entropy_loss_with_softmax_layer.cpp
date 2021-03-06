#include "layer/cross_entropy_loss_with_softmax_layer.h"
#include <cfloat>

#include "test_utils.hpp"

namespace mkt {
    CrossEntropyLossWithSoftmaxLayer::CrossEntropyLossWithSoftmaxLayer(
        Layer* prevLayer,
        std::string id): Layer(LayerType::CROSS_LOSS_SOFTMAX)
    {
        batchSize_ = prevLayer->pDst_->getNumOfData();
        pPrevLayer_ = prevLayer;

        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

        /**************************************************
         * Now i just fix the input height and width is 1
         * that means the previous layer must be DenseLayer
         **************************************************/
        oh_ = 1;
        ow_ = 1;
        oc_ = 1;


        /*
         * Setting SoftmaxLayer:
         * Use SoftmaxLayer as data preprocessor to
         * calculate probability.then calculate loss
         */
        softmaxLayer_.pPrevLayer_ = prevLayer;
        softmaxLayer_.Reshape(batchSize_, ih, iw, ic);

        pDst_ = new Tensor{1, 1, 1, 1}; // only for loss

        /*
         * [pgDst_]
         * Because the loss layer is the last layer of net
         * there is no need for gradient data from next layer
         */

        /*
         * pLabel_ is a Tensor for storing Label data
         * For now, the label data of Cross Entroyp Loss
         * is Scale Not One-Hot encoding vector
         */
        pLabel_ = new Tensor{batchSize_, 1, 1, 1};

    }

    // Destructor
    CrossEntropyLossWithSoftmaxLayer::~CrossEntropyLossWithSoftmaxLayer() {

        delete pLabel_;

    }

    // Initialization
    void CrossEntropyLossWithSoftmaxLayer::initialize(NetMode mode) {

        MKT_Assert(pDst_ != nullptr, "pDst_ is null");
        MKT_Assert(pLabel_ != nullptr, "pLabel_ is null");

        initOutputTensor();

        pLabel_->allocate();
    }

    /*********************************************************************
     * Each Turth Label is a "scalar value" not "One-Hot encoding vector"
     * For example:
     * There are 10 classes and batch size = 64, so there are 64 labels
     *  the range of each label is 0 ~ 9
     *********************************************************************/
    void CrossEntropyLossWithSoftmaxLayer::LoadLabel(int num, const int* pLabel) {

        CHECK_EQ(pLabel_->getWholeSize(), num, __FILE__);

        int numClass = softmaxLayer_.pDst_->getChannel();

        float* pLabelData = pLabel_->getCPUData();

        for (int i = 0; i < num; ++i)
        {
            CHECK_LT(pLabel[i], numClass, __FILE__, __LINE__);
            pLabelData[i] = static_cast<float>( pLabel[i] );
        }
    }

    void CrossEntropyLossWithSoftmaxLayer::LoadLabel(const std::vector<int>& labels) {

        CHECK_EQ(pLabel_->getWholeSize(), static_cast<int>(labels.size()), __FILE__);

        int numClass = softmaxLayer_.pDst_->getChannel();

        float* pLabelData = pLabel_->getCPUData();

        for (size_t i = 0; i < labels.size(); ++i)
        {
            CHECK_LT(labels[i], numClass, __FILE__, __LINE__);
            pLabelData[i] = static_cast<float>( labels[i] );
        }
    }

    /*****************************************************
     * Cross Entropy Loss = - SUM[ y_i_hat * ln(y_i) ]
     * y_i_hat: truth label of class i
     * y_i    : predict probability of class i
     *
     * Loss = - ln(y_k) where k is the index of y_i_hat = 1
     *
     *****************************************************/
    void CrossEntropyLossWithSoftmaxLayer::Forward() {

        // Reset Data
        // pDst_->resetData();

        Tensor* pSrc = pPrevLayer_->pDst_;

        softmaxLayer_.Forward();

        int dim = softmaxLayer_.pDst_->getSize3D();
        int size2D = softmaxLayer_.pDst_->getSize2D();
        int numClass = softmaxLayer_.pDst_->getChannel();

        const float* pTruth_label = pLabel_->getCPUData();
        const float* prob_data = softmaxLayer_.pDst_->getCPUData();
        float loss = 0;

        for (int b = 0; b < batchSize_; ++b)
        {
            for (int i = 0; i < size2D; ++i)
            {
                const int label_value = static_cast<int>( pTruth_label[i + b*size2D]);
                // fprintf(stderr, "label_value[%d]: %d\n", (i + b*size2D), label_value);
                CHECK_GE(label_value, 0, __func__);
                CHECK_LT(label_value, numClass, __func__);
                // fprintf(stderr, "i: %d, label_value: %d, size2D: %d, b: %d, dim: %d\n", i, label_value, size2D, b, dim);
                // fprintf(stderr, "prob_data[%d]: %f\n", i + label_value*size2D + b*dim, prob_data[i + label_value*size2D + b*dim]);
                loss -= log( std::max(prob_data[i + label_value*size2D + b*dim], FLT_MIN) );
                // fprintf(stderr, "[cross entropy loss forwartd]: prob_data[%d] = %f\n", i + label_value*size2D + b*dim, prob_data[i + label_value*size2D + b*dim]);
            }
        }
        // fprintf(stderr, "loss - %f\n", loss);
        pDst_->getCPUData()[0] = loss / batchSize_ * size2D;
        // fprintf(stderr, "%s:%d loss: %f\n", __FILE__, __LINE__, pDst_->getCPUData()[0]);
    }


    /****************************************************
     * dloss/dx_j = exp(x_j) / sum(exp(x_i)) - 1, if j = k
     *            = exp(x_j) / sum(exp(x_i))    , if j != k
     * k is the index of label which equals to 1
     ****************************************************/
    void CrossEntropyLossWithSoftmaxLayer::Backward() {

        int wholeSize = softmaxLayer_.pDst_->getWholeSize();
        int dim = softmaxLayer_.pDst_->getSize3D();
        int size2D = softmaxLayer_.pDst_->getSize2D();
        float* prob = softmaxLayer_.pDst_->getCPUData();

        float* pSrc_dif = pPrevLayer_->pgDst_->getCPUData();

        if (pSrc_dif)
        {
            // pSrc_dif = prob = exp(xj) / sum(exp(xi))
            mem_copy_cpu(wholeSize, prob, pSrc_dif);

            const float* pTruth_label = pLabel_->getCPUData();
            for (int b = 0; b < batchSize_; ++b)
            {
                for (int i = 0; i < size2D; ++i)
                {
                    // pSrc_dif[j] = exp(xj) / sum(exp(xi)) - 1, if j = the index of y = 1
                    const int label_value = static_cast<int>( pTruth_label[i + b*size2D]);
                    pSrc_dif[i + label_value*size2D + b*dim] -= 1;
                }
            }
        }
    }
}

