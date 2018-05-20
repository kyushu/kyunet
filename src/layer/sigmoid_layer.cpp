#include "layer/sigmoid_layer.h"

namespace mkt {

    template<typename T>
    SigmoidLayer<T>::SigmoidLayer(
        Layer<T>* prevLayer,
        std::string id
    ): Layer<T>(LayerType::SIGMOID)
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

        this->pDst_  = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_};
        this->pgDst_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_};

    };

    template<typename T>
    SigmoidLayer<T>::~SigmoidLayer() {};

    template<typename T>
    void SigmoidLayer<T>::initialize(NetMode mode) {
        MKT_Assert( this->pDst_ != nullptr, "pDst_ is null" );
        MKT_Assert( this->pgDst_ != nullptr, "pgDst_ is null" );

        this->initOutputTensor();
        this->initGradTensor();
    };

    template<typename T>
    void SigmoidLayer<T>::Forward() {

        Tensor<T>* pSrc = this->pPrevLayer_->pDst_;

        sigmoid_act_.Forward(pSrc, this->pDst_);
    };

    template<typename T>
    void SigmoidLayer<T>::Backward() {
        fprintf(stderr, "%s: %s: %d\n", __FILE__, __func__, __LINE__);
    };

    // Explicitly instantiate the template, and its member definitions
    template class SigmoidLayer<float>;

} // namespace mkt
