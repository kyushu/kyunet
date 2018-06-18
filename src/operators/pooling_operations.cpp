
#include <limits>
#include "operations/pooling_operations.h"


namespace mkt {
namespace op {

    template<typename T>
    void pooling (
        const int numOfSample, const PoolingMethodType poolingType, const ConvParam& convParam,
        const int fh, const int fw,
        Tensor<T>* pSrc, Tensor<T>* pDst, Tensor<T>* pMask
        )
    {

        T* pSrcData = pSrc->getCPUData();
        int src_size2d = pSrc->getSize2D();
        int ih = pSrc->getHeight();
        int iw = pSrc->getWidth();

        T* pDstData = pDst->getCPUData();
        int oh = pDst->getHeight();
        int ow = pDst->getWidth();
        int oc = pDst->getChannel();
        int dst_size2D = pDst->getSize2D();


        switch (poolingType) {
            case PoolingMethodType::MAX:
            {
                T* pMaskData = pMask->getCPUData();

                for (int b = 0; b < numOfSample; ++b) {
                    for (int c = 0; c < oc; ++c) {
                        for (int ph = 0; ph < oh; ++ph) {
                            for (int pw = 0; pw < ow; ++pw) {
                                int hstart = ph * convParam.stride_h_ - convParam.pad_h_;
                                int wstart = pw * convParam.stride_w_ - convParam.pad_w_;
                                int hend = std::min(hstart + fh, ih);
                                int wend = std::min(wstart + fw, iw);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);

                                int pool_index = ph * ow + pw;
                                // float fmax = -FLT_MAX;
                                T fmax = -std::numeric_limits<T>::infinity();

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
                for (int b = 0; b < numOfSample; ++b) {
                    for (int c = 0; c < oc; ++c) {
                        for (int ph = 0; ph < oh; ++ph) {
                            for (int pw = 0; pw < ow; ++pw) {
                                int hstart = ph * convParam.stride_h_ - convParam.pad_h_;
                                int wstart = pw * convParam.stride_w_ - convParam.pad_w_;

                                // we need to calculate the size of pooling window for average
                                // so we need to take care of "source size + padding" for the
                                // boundary of hend and wend
                                int hend = std::min(hstart + fh, ih+convParam.pad_h_);
                                int wend = std::min(wstart + fw, iw+convParam.pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);
                                // but we don't need to calculate the value at position ih+pad_h_ and iw+pad_w_
                                hend = std::min(hend, ih);
                                wend = std::min(wend, iw);

                                int pool_index = ph * ow + pw;
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
    void pooling_gradient(const int num_sample, const PoolingMethodType poolingType, const ConvParam& convParam,
        const int fh, const int fw,
        Tensor<T>* pgSrc, Tensor<T>* pgDst , Tensor<T>* pMaxMask
        )
    {
        T* pgDstData = pgDst->getCPUData();
        int dst_size2D = pgDst->getSize2D();
        int oh = pgDst->getHeight();
        int ow = pgDst->getWidth();
        int oc = pgDst->getChannel();

        T* pgSrcData = pgSrc->getCPUData();
        int ih = pgSrc->getHeight();
        int iw = pgSrc->getWidth();
        int src_size2d = pgSrc->getSize2D();

        switch(poolingType) {
            case PoolingMethodType::MAX:
            {
                T* pMaskData = pMaxMask->getCPUData();

                for (int b = 0; b < num_sample; ++b) {
                    for (int c = 0; c < oc; ++c) {
                        // Pooling window
                        for (int ph = 0; ph < oh; ++ph) {
                            for (int pw = 0; pw < ow; ++pw) {
                                int pool_index = ph * ow + pw;
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
                 for (int b = 0; b < num_sample; ++b) {
                    for (int c = 0; c < oc; ++c) {
                        // Pooling window
                        for (int ph = 0; ph < oh; ++ph) {
                            for (int pw = 0; pw < ow; ++pw) {
                                int hstart = ph * convParam.stride_h_ - convParam.pad_h_;
                                int wstart = pw * convParam.stride_w_ - convParam.pad_w_;
                                int hend = std::min(hstart + fh, ih+convParam.pad_h_);
                                int wend = std::min(wstart + fw, iw+convParam.pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);
                                hend = std::min(hend, ih);
                                wend = std::min(wend, iw);

                                for (int h = hstart; h < hend; ++h){
                                    for (int w = wstart; w < wend; ++w){
                                        pgSrcData[h*iw + w] += pgDstData[ph*ow + pw] / pool_size;
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


    // Explicit instantiation
    template void pooling<float>(
        const int , const PoolingMethodType , const ConvParam& ,
        const int , const int ,
        Tensor<float>* , Tensor<float>* , Tensor<float>* );

    template void pooling_gradient<float>(
        const int , const PoolingMethodType , const ConvParam& ,
        const int , const int ,
        Tensor<float>* , Tensor<float>*  , Tensor<float>* );

} // namespace op
} // namespace mkt
