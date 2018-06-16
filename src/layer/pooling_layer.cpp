#include "layer/pooling_layer.h"
#include <cfloat>


namespace mkt {

    template<typename T>
    PoolingLayer<T>::PoolingLayer(
        Layer<T>* prevLayer,
        std::string id,
        int fh,
        int fw,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        PoolingMethodType type
    ):
        fh_{fh},  fw_{fw},
        stride_h_{stride_h}, stride_w_{stride_w},
        pad_h_{pad_h}, pad_w_{pad_w},
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
        this->ow_ = static_cast<int>( ceil( static_cast<float>(iw - fw_ + 2*pad_w_) / stride_w_ ) ) + 1;
        this->oh_ = static_cast<int>( ceil( static_cast<float>(ih - fh_ + 2*pad_h_) / stride_h_ ) ) + 1;

        if (pad_h_ || pad_w_)
        {
            if (( this->ow_ - 1 ) * stride_w_ > iw + pad_w_) { --this->ow_; }
            if (( this->oh_ - 1 ) * stride_h_ > ih + pad_h_) { --this->oh_; }
        }

        MKT_Assert(( this->ow_-1)*stride_w_ < iw + pad_w_, "polling size");
        MKT_Assert(( this->oh_-1)*stride_h_ < ih + pad_h_, "polling size");

        this->pDst_  = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_};
        this->pgDst_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_};

        if (type_ == PoolingMethodType::MAX)
        {
            // For storing index of max value of src data in each pooling window
            pMask = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };
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

        stride_h_ = params.stride_h;
        stride_w_ = params.stride_w;

        pad_h_ = params.pad_h;
        pad_w_ = params.pad_w;

        this->oc_ = ic;
        // calculate pDst size: ow = (W-f +2p)/s +1
        this->ow_ = static_cast<int>( ceil( static_cast<float>(iw - fw_ + 2*pad_w_) / stride_w_ ) ) + 1;
        this->oh_ = static_cast<int>( ceil( static_cast<float>(ih - fh_ + 2*pad_h_) / stride_h_ ) ) + 1;

        if (pad_h_ || pad_w_)
        {
            if (( this->ow_ - 1 ) * stride_w_ > iw + pad_w_) { --this->ow_; }
            if (( this->oh_ - 1 ) * stride_h_ > ih + pad_h_) { --this->oh_; }
        }

        MKT_Assert(( this->ow_-1 )*stride_w_ < iw + pad_w_, "polling size");
        MKT_Assert(( this->oh_-1 )*stride_h_ < ih + pad_h_, "polling size");

        this->pDst_  = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };
        this->pgDst_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };

        if (type_ == PoolingMethodType::MAX)
        {
            // For storing index of max value of src data in each pooling window
            pMask = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_};
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
            MKT_Assert(pMask != nullptr, "pMask is null");
            pMask->allocate();
        }
    }

    // Computation Function
    template<typename T>
    void PoolingLayer<T>::Forward() {

        // Reset data
        this->pDst_->resetData();
        this->pgDst_->resetData();

        Tensor<T>* pSrc = this->pPrevLayer_->pDst_;
        T* pSrcData = pSrc->getCPUData();
        int src_size2d = pSrc->getSize2D();
        int ih = pSrc->getHeight();
        int iw = pSrc->getWidth();

        T* pDstData = this->pDst_->getCPUData();
        int dst_size2D = this->pDst_->getSize2D();

        switch (type_) {
            case PoolingMethodType::MAX:
            {
                T* pMaskData = pMask->getCPUData();

                for (int b = 0; b < this->batchSize_; ++b) {
                    for (int c = 0; c < this->oc_; ++c) {
                        for (int ph = 0; ph < this->oh_; ++ph) {
                            for (int pw = 0; pw < this->ow_; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = std::min(hstart + fh_, ih);
                                int wend = std::min(wstart + fw_, iw);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);

                                int pool_index = ph * this->ow_ + pw;
                                float fmax = -FLT_MAX;

                                // pooling window
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
                                        int index = h * iw + w;

                                        // fmax = pSrcData[index] > fmax ? pSrcData[index] : fmax;
                                        if (pSrcData[index] > fmax)
                                        {
                                            fmax = pSrcData[index];
                                            // pool_index is in dst_size2D
                                            // index      is in src_size2d
                                            pMaskData[pool_index] = index;
                                        }
                                    }
                                }
                                pDstData[pool_index] = fmax;
                            }
                        }
                        // offset to next channel
                        pDstData += dst_size2D;
                        pMaskData += dst_size2D;
                        pSrcData += src_size2d;
                    }
                }
            }
                break;
            case PoolingMethodType::AVG:
                for (int b = 0; b < this->batchSize_; ++b) {
                    for (int c = 0; c < this->oc_; ++c) {
                        for (int ph = 0; ph < this->oh_; ++ph) {
                            for (int pw = 0; pw < this->ow_; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;

                                // we need to calculate the size of pooling window for average
                                // so we need to take care of "source size + padding" for the
                                // boundary of hend and wend
                                int hend = std::min(hstart + fh_, ih+pad_h_);
                                int wend = std::min(wstart + fw_, iw+pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);
                                // but we don't need to calculate the value at position ih+pad_h_ and iw+pad_w_
                                hend = std::min(hend, ih);
                                wend = std::min(wend, iw);

                                int pool_index = ph * this->ow_ + pw;
                                T favg = 0;
                                for (int h = hstart; h < hend; ++h){
                                    for (int w = wstart; w < wend; ++w){
                                        int index = h * iw + w;
                                        favg += pSrcData[index];
                                    }
                                }

                                pDstData[pool_index] = favg / pool_size;
                            }
                        }
                        // offset to next channel
                        pDstData += dst_size2D;
                        pSrcData += src_size2d;
                    }
                }
                break;
            default:
                fprintf(stderr, "wrong pooling method\n");
                break;
        }
    }

    template<typename T>
    void PoolingLayer<T>::Backward() {

        T* pgDstData = this->pgDst_->getCPUData();
        int dst_size2D = this->pgDst_->getSize2D();
        T* pgSrcData = this->pPrevLayer_->pgDst_->getCPUData();
        int ih = this->pPrevLayer_->pgDst_->getHeight();
        int iw = this->pPrevLayer_->pgDst_->getWidth();
        int src_size2d = this->pPrevLayer_->pgDst_->getSize2D();

        switch(type_) {
            case PoolingMethodType::MAX:
            {
                T* pMaskData = pMask->getCPUData();

                for (int b = 0; b < this->batchSize_; ++b) {
                    for (int c = 0; c < this->oc_; ++c) {
                        // Pooling window
                        for (int ph = 0; ph < this->oh_; ++ph) {
                            for (int pw = 0; pw < this->ow_; ++pw) {
                                int pool_index = ph * this->ow_ + pw;
                                int src_index = pMaskData[pool_index];
                                pgSrcData[src_index] += pgDstData[pool_index];
                            }
                        }

                        // offset to next channel
                        pMaskData += dst_size2D;
                        pgDstData += dst_size2D;
                        pgSrcData += src_size2d;

                    }
                }
            }
                break;
            case PoolingMethodType::AVG:
                 for (int b = 0; b < this->batchSize_; ++b) {
                    for (int c = 0; c < this->oc_; ++c) {
                        // Pooling window
                        for (int ph = 0; ph < this->oh_; ++ph) {
                            for (int pw = 0; pw < this->ow_; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = std::min(hstart + fh_, ih+pad_h_);
                                int wend = std::min(wstart + fw_, iw+pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);
                                hend = std::min(hend, ih);
                                wend = std::min(wend, iw);

                                for (int h = hstart; h < hend; ++h){
                                    for (int w = wstart; w < wend; ++w){
                                        pgSrcData[h*iw + w] += pgDstData[ph*this->ow_ + pw] / pool_size;
                                    }
                                }
                            }
                        }
                        pgDstData += dst_size2D;
                        pgSrcData += src_size2d;
                    }
                }
                break;
            default:
                fprintf(stderr, "wrong pooling method\n");
        }
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
