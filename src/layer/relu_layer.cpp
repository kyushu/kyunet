#include "layer/relu_layer.h"

namespace mkt {

    template<typename T>
    ReluLayer<T>::ReluLayer(
        Layer<T>* prevLayer,
        std::string id,
        T negative_slope
    ): negative_slope_{negative_slope}, Layer<T>(LayerType::RELU)
    {

        this->id_ = id;

        this->batchSize_ = prevLayer->pDst_->getNumOfData();

        this->pPrevLayer_ = prevLayer;

        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

        this->oh_ = ih;
        this->ow_ = iw;
        this->oc_ = ic;

        this->pDst_  = new Tensor<T>{ this->batchSize_, this->oc_, this->oh_, this->ow_ };
        this->pgDst_ = new Tensor<T>{ this->batchSize_, this->oc_, this->oh_, this->ow_ };

    };

    template<typename T>
    ReluLayer<T>::~ReluLayer() {};

    template<typename T>
    void ReluLayer<T>::initialize(NetMode mode) {

        MKT_Assert( this->pDst_ != nullptr, "pDst_ is null" );
        MKT_Assert( this->pgDst_ != nullptr, "pgDst_ is null" );

        this->initOutputTensor();
        this->initGradTensor();
    };

    template<typename T>
    void ReluLayer<T>::Forward() {
        Tensor<T>* pSrc = this->pPrevLayer_->pDst_;

        relu_act_.Forward(pSrc, this->pDst_);
    };

    template<typename T>
    void ReluLayer<T>::Backward() {
        relu_act_.Backward(this->pDst_, this->pgDst_, this->pgDst_);
    };

    // Explicitly instantiate the template, and its member definitions
    template class ReluLayer<float>;

} // namespace mkt
