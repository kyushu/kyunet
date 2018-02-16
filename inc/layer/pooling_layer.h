#ifndef _POOL_LAYER_H_
#define _POOL_LAYER_H_

#include "layer.h"

namespace mkt {

    class PoolingLayer: public Layer
    {
    public:
        int fh_; // filter height
        int fw_; // filter width
        int stride_h_;
        int stride_w_;
        int pad_h_;
        int pad_w_;
        PoolingMethodType type_;


        PoolingLayer(
            Layer* prevLayer,
            std::string id,
            int kernel_Height,
            int kernel_width,
            int stride_h,
            int stride_w,
            int pad_h,
            int pad_w,
            PoolingMethodType type
            );
        ~PoolingLayer();

        void initialize();

        // Computation Function
        void forward();
        void backward();

        // Getter Function
        int getFilterHeight();
        int getFilterWidth();
    };

}

#endif