
#include "layer/batchNorm.h"

namespace mkt {

    template<typename T>
    BatchNorm<T>::BatchNorm(Layer<T>* prevLayer, std::string id):Layer<T>(LayerType::BATCHNORM)
    {

        this->id_ = id;
        this->batchSize_ = prevLayer->pDst_->getNumOfData();
        this->pPrevLayer_ = prevLayer;

        this->oc_ = this->pPrevLayer_->pDst_->getChannel();
        this->oh_ = this->pPrevLayer_->pDst_->getChannel();
        this->ow_ = this->pPrevLayer_->pDst_->getWidth();


        // The dimemsion of output is same as previous output
        this->pDst_ = new Tensor<T>{this->batchSize_, this->oh_, this->ow_, this->oc_};
        this->pgDst_ = new Tensor<T>{this->batchSize_, this->oh_, this->ow_, this->oc_};

        // pW_ = gamma
        this->pW_   = new Tensor<T>{1, 1, 1, this->oc_};
        this->pgW_  = new Tensor<T>{1, 1, 1, this->oc_};

        // pB_ = beta
        this->pB_   = new Tensor<T>{1, 1, 1, this->oc_};
        this->pgB_  = new Tensor<T>{1, 1, 1, this->oc_};


        // For current batch data
        pMu_        = new Tensor<T>{1, 1, 1, this->oc_};
        pVariance_  = new Tensor<T>{1, 1, 1, this->oc_};

        // Set initial value of gamma = 1
        this->weightInitType_ = InitializerType::ONE;
        // Set initial value of beta = 0
        this->biasInitType_ = InitializerType::ZERO;
    }


    template<typename T>
    void BatchNorm<T>::initialize(NetMode mode)
    {
        MKT_Assert(this->pDst_  != nullptr, "pDst_ is null");
        MKT_Assert(this->pgDst_ != nullptr, "pgDst_ is null");
        MKT_Assert(this->pW_    != nullptr, "pW_ is null");
        MKT_Assert(this->pgW_   != nullptr, "pgW_ is null");
        MKT_Assert(this->pB_    != nullptr, "pB_ is null");
        MKT_Assert(this->pgB_   != nullptr, "pgB_ is null");

        this->initOutputTensor();
        this->initWeightTensor();
        this->initBiasTensor();

        this->initGradTensor();
        this->initGradWeightTensor();
        this->initGradBiasTensor();

        pMu_->allocate();
        std::fill_n(pMu_->getCPUData(), pMu_->getWholeSize(), 0);
        pVariance_->allocate();
        std::fill_n(pVariance_->getCPUData(), pVariance_->getWholeSize(), 0);
    }

    template<typename T>
    void BatchNorm<T>::Forward()
    {

    }

    template<typename T>
    void BatchNorm<T>::Backward()
    {

    }



    template class BatchNorm<float>;
}
