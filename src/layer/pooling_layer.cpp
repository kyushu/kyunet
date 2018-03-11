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

        pPrevLayer_ = prevLayer;

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

        pDst_  = new Tensor{batchSize_, oh_, ow_, oc_};
        pgDst_ = new Tensor{batchSize_, oh_, ow_, oc_};

        if (type_ == PoolingMethodType::MAX)
        {
            // For storing index of max value of src data in each pooling window
            pMask = new Tensor{batchSize_, oh_, ow_, oc_};
        }
    }

    // Destructor
    PoolingLayer::~PoolingLayer() {

    }


    // Initialization
    void PoolingLayer::initialize() {

        initOutputTensor();

        pMask->allocate();
    }

    // Computation Function
    void PoolingLayer::Forward() {

        Tensor* pSrc = pPrevLayer_->pDst_;

        float* pSrcData = pSrc->cpu_data();
        int ih = pSrc->Height();
        int iw = pSrc->Width();

        float* pDstData = pDst_->cpu_data();
        int dst_size2D = pDst_->Size2D();

        float* pMaskData = pMask->cpu_data();


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

                                int pool_index = ph * ow_ + pw;
                                float favg = 0;
                                for (int h = hstart; h < hend; ++h){
                                    for (int w = wstart; w < wend; ++w){
                                        int index = h * iw + w;
                                        favg += pSrcData[index];
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
        float* pMaskData = pMask->cpu_data();
        float* pgDstData = pgDst_->cpu_data();
        int dst_size2D = pgDst_->Size2D();

        float* pgSrcData = pPrevLayer_->pgDst_->cpu_data();
        int ih = pPrevLayer_->pgDst_->Height();
        int iw = pPrevLayer_->pgDst_->Width();
        int src_size2d = pPrevLayer_->pgDst_->Size2D();

        switch(type_) {
            case PoolingMethodType::MAX:
                for (int b = 0; b < batchSize_; ++b) {
                    for (int c = 0; c < oc_; ++c) {
                        // Pooling window
                        for (int ph = 0; ph < oh_; ++ph) {
                            for (int pw = 0; pw < ow_; ++pw) {
                                int pool_index = ph * ow_ + pw;
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
                break;
            case PoolingMethodType::AVG:
                 for (int b = 0; b < batchSize_; ++b) {
                    for (int c = 0; c < oc_; ++c) {
                        // Pooling window
                        for (int ph = 0; ph < oh_; ++ph) {
                            for (int pw = 0; pw < ow_; ++pw) {
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
                                        pgSrcData[h*iw + w] += pgDstData[ph*ow_ + pw] / pool_size;
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

    /*


  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  */



    // Getter Function
    int PoolingLayer::getFilterHeight() {
        return fh_;
    }
    int PoolingLayer::getFilterWidth() {
        return fw_;
    }


}
