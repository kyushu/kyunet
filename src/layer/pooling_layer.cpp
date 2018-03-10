#include "layer/pooling_layer.h"
#include <cfloat>

namespace mkt {

    PoolingLayer::PoolingLayer(
        Layer* prevLayer,
        std::string id,
        int kernel_Height,
        int kernel_width,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        PoolingMethodType type
    ):
        fh_{kernel_Height},
        fw_{kernel_width},
        stride_h_{stride_h},
        stride_w_{stride_w},
        pad_h_{pad_h},
        pad_w_{pad_w},
        type_{type},
        Layer(LayerType::Pooling)
    {
        id_ = id;

        batchSize_ = prevLayer->pDst_->NumOfData();
        int ih = prevLayer->pDst_->Height();
        int iw = prevLayer->pDst_->Width();
        int ic = prevLayer->pDst_->Channel();
        pSrc_ = prevLayer->pDst_;

        oc_ = ic;
        // calculate pDst size: ow = (W-f +2p)/s +1
        ow_ = static_cast<int>( ceil( static_cast<float>(iw - fw_ + 2*pad_w_) / stride_w ) ) + 1;
        oh_ = static_cast<int>( ceil( static_cast<float>(ih - fh_ + 2*pad_h_) / stride_h ) ) + 1;

        if (pad_h_ || pad_w_)
        {
            if ((ow_ - 1) * stride_w > iw + pad_w_) { --ow_; }
            if ((oh_ - 1) * stride_h > ih + pad_h_) { --oh_; }
        }

        MKT_Assert((ow_-1)*stride_w_ < iw + pad_w_, "polling size");
        MKT_Assert((oh_-1)*stride_h_ < ih + pad_h_, "polling size");

        pDst_ = new Tensor{batchSize_, oh_, ow_, oc_};
    }

    // Destructor
    PoolingLayer::~PoolingLayer() {

    }


    // Initialization
    void PoolingLayer::initialize() {

        initOutputTensor();

    }

    // Computation Function
    void PoolingLayer::Forward() {

        float* pSrcData = pSrc_->cpu_data();
        float* pDstData = pDst_->cpu_data();
        int ih = pSrc_->Height();
        int iw = pSrc_->Width();

        switch (type_) {
            case PoolingMethodType::MAX:
                for (int b = 0; b < batchSize_; ++b) {
                    for (int c = 0; c < oc_; ++c) {
                        for (int ph = 0; ph < oh_; ++ph) {
                            for (int pw = 0; pw < ow_; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = std::min(hstart + fh_, ih);
                                int wend = std::min(wstart + fw_, iw);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);

                                int pool_index = ph * ow_ + pw;
                                float fmax = -FLT_MAX;

                                // pooling window
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
                                        int index = h * iw + w;
                                        fmax = pSrcData[index] > fmax ? pSrcData[index] : fmax;
                                    }
                                }
                                pDstData[pool_index] = fmax;
                            }
                        }
                    }
                }
                break;
            case PoolingMethodType::AVG:
                for (int b = 0; b < batchSize_; ++b) {
                    for (int c = 0; c < oc_; ++c) {
                        for (int ph = 0; ph < oh_; ++ph) {
                            for (int pw = 0; pw < ow_; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;

                                // Because we need to calculate the size of pooling window
                                // we need to take care of "source size + padding" for the
                                // boundary of hend and wend
                                int hend = std::min(hstart + fh_, ih+pad_h_);
                                int wend = std::min(wstart + fw_, iw+pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);
                                // but we don't need to calculate the value at position ih+pad_h_ and iw+pad_w_
                                hend = std::min(hend, ih);
                                wend = std::min(wend, iw);

                                int pool_index = ph * ow_ + pw;
                                float favg = 0;
                                // fprintf(stderr, "pool_index: %d\n", pool_index);
                                // fprintf(stderr, "h: %d - %d\n", hstart, hend);
                                // fprintf(stderr, "w: %d - %d\n", wstart, wend);
                                for (int h = hstart; h < hend; ++h){
                                    for (int w = wstart; w < wend; ++w){
                                        int index = h * iw + w;
                                        // fprintf(stderr, "[%d]=%.3f\n", index, pSrcData[index]);
                                        favg += pSrcData[index];
                                        // fprintf(stderr, "favg: %.3f\n", favg);
                                    }
                                }

                                pDstData[pool_index] = favg / pool_size;
                            }
                        }
                    }
                }
                break;
            default:
                fprintf(stderr, "wrong pooling method\n");
                break;
        }
    }

    void PoolingLayer::Backward() {

    }

    // Getter Function
    int PoolingLayer::getFilterHeight() {
        return fh_;
    }
    int PoolingLayer::getFilterWidth() {
        return fw_;
    }


}
