#include "layer/pooling_layer.h"
#include <cfloat>

#include "operations/conv_operations.h"
#include "operations/pooling_operations.h"

namespace mkt {

    template<typename T>
    PoolingLayer<T>::PoolingLayer(
        Layer<T>* prevLayer,
        std::string id,
        int fh,
        int fw,
        // int stride_h,
        // int stride_w,
        // int pad_h,
        // int pad_w,
        ConvParam convParam,
        PoolingMethodType type
    ):
        fh_{fh},  fw_{fw},
        // stride_h_{stride_h}, stride_w_{stride_w},
        // pad_h_{pad_h}, pad_w_{pad_w},
        convParam_{convParam},
        type_{type},
        Layer<T>(LayerType::POOLING)
    {
        this->id_ = id;

        this->batchSize_ = prevLayer->pDst_->getNumOfData();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

        this->pPrevLayer_ = prevLayer;

        this->oc_ = ic;
        // calculate pDst size: ow = (W-f +2p)/s +1
        // this->ow_ = static_cast<int>( ceil( static_cast<float>(iw - fw_ + 2*pad_w_) / stride_w_ ) ) + 1;
        // this->oh_ = static_cast<int>( ceil( static_cast<float>(ih - fh_ + 2*pad_h_) / stride_h_ ) ) + 1;

        // if (pad_h_ || pad_w_)
        // {
        //     if (( this->ow_ - 1 ) * stride_w_ > iw + pad_w_) { --this->ow_; }
        //     if (( this->oh_ - 1 ) * stride_h_ > ih + pad_h_) { --this->oh_; }
        // }

        // MKT_Assert(( this->ow_-1)*stride_w_ < iw + pad_w_, "polling size");
        // MKT_Assert(( this->oh_-1)*stride_h_ < ih + pad_h_, "polling size");

        op::calcConvOutputSize(iw, ih, fw_, fh_, convParam_, this->ow_, this->oh_);
        this->pDst_  = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_};
        this->pgDst_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_};

        if (type_ == PoolingMethodType::MAX)
        {
            // For storing index of max value of src data in each pooling window
            pMask_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };
        }
    }

    template<typename T>
    PoolingLayer<T>::PoolingLayer(Layer<T>* prevLayer, std::string id, LayerParams params):Layer<T>(LayerType::POOLING) {

        this->id_ = id;

        this->batchSize_ = prevLayer->pDst_->getNumOfData();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

        this->pPrevLayer_ = prevLayer;

        // Parameter setting
        type_ = params.pooling_type;
        fh_ = params.fh;
        fw_ = params.fw;

        // stride_h_ = params.stride_h;
        // stride_w_ = params.stride_w;

        // pad_h_ = params.pad_h;
        // pad_w_ = params.pad_w;
        convParam_.stride_h_ = params.stride_h;
        convParam_.stride_w_ = params.stride_w;

        convParam_.paddingType_ = params.padding_type;
        convParam_.pad_h_ = params.pad_h;
        convParam_.pad_w_ = params.pad_w;

        // Temporary set dilation to 1
        convParam_.dilation_h_ = params.dilation_h;
        convParam_.dilation_w_ = params.dilation_w;

        this->oc_ = ic;
        // // calculate pDst size: ow = (W-f +2p)/s +1
        // this->ow_ = static_cast<int>( ceil( static_cast<float>(iw - fw_ + 2*pad_w_) / stride_w_ ) ) + 1;
        // this->oh_ = static_cast<int>( ceil( static_cast<float>(ih - fh_ + 2*pad_h_) / stride_h_ ) ) + 1;

        // if (pad_h_ || pad_w_)
        // {
        //     if (( this->ow_ - 1 ) * stride_w_ > iw + pad_w_) { --this->ow_; }
        //     if (( this->oh_ - 1 ) * stride_h_ > ih + pad_h_) { --this->oh_; }
        // }

        // MKT_Assert(( this->ow_-1 )*stride_w_ < iw + pad_w_, "polling size");
        // MKT_Assert(( this->oh_-1 )*stride_h_ < ih + pad_h_, "polling size");


        op::calcConvOutputSize(iw, ih, fw_, fh_, convParam_, this->ow_, this->oh_);
        this->pDst_  = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };
        this->pgDst_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };

        if (type_ == PoolingMethodType::MAX)
        {
            // For storing index of max value of src data in each pooling window
            pMask_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_};
        }
    }

    // Destructor
    template<typename T>
    PoolingLayer<T>::~PoolingLayer() {

    }


    // Initialization
    template<typename T>
    void PoolingLayer<T>::initialize(NetMode mode) {

        MKT_Assert( this->pDst_ != nullptr, "pDst_ is null");
        MKT_Assert( this->pgDst_ != nullptr, "pgDst_ is null");

        this->initOutputTensor();
        this->initGradTensor();

        if (type_ == PoolingMethodType::MAX)
        {
            MKT_Assert(pMask_ != nullptr, "pMask is null");
            pMask_->allocate();
        }
    }

    // Computation Function
    template<typename T>
    void PoolingLayer<T>::Forward() {

        // Reset data
        this->pDst_->resetData();
        this->pgDst_->resetData();

        op::pooling<T>(this->batchSize_, type_, convParam_, fh_, fw_, this->pPrevLayer_->pDst_, this->pDst_, pMask_);

    }

    template<typename T>
    void PoolingLayer<T>::Backward() {

        op::pooling_gradient<T>(this->batchSize_, type_, convParam_,
        fh_, fw_,
        this->pPrevLayer_->pgDst_, this->pgDst_, pMask_
        );

    }

    // Getter Function
    template<typename T>
    int PoolingLayer<T>::getFiltergetHeight() {
        return fh_;
    }
    template<typename T>
    int PoolingLayer<T>::getFiltergetWidth() {
        return fw_;
    }

    // Explicitly instantiate the template, and its member definitions
    template class PoolingLayer<float>;

} // namespace mkt
