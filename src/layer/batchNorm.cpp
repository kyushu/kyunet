
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


        // The dimemsion of output is same as previous
        this->pDst_ = new Tensor<T>{this->batchSize_, this->oh_, this->ow_, this->oc_};
        this->pgDst_ = new Tensor<T>{this->batchSize_, this->oh_, this->ow_, this->oc_};

        // this->pW_   = new Tensor<T>{ic, fh_, fw_, fc_};
        // this->pgW_  = new Tensor<T>{ic, fh_, fw_, fc_};

        // this->pB_   = new Tensor<T>{1, 1, 1, this->oc_};
        // this->pgB_  = new Tensor<T>{1, 1, 1, this->oc_};

    }



    template class BatchNorm<float>;
}
