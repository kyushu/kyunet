#ifndef MKT_POOL_LAYER_H
#define MKT_POOL_LAYER_H

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

        Tensor* pMask;


        PoolingLayer(Layer* prevLayer, std::string id,
            int kernel_Height, int kernel_width,
            int stride_h, int stride_w,
            int pad_h, int pad_w,
            PoolingMethodType type
            );
        PoolingLayer(Layer* prevLayer, std::string id, LayerParams params);

        ~PoolingLayer();

        void initialize();

        // Computation Function
        void Forward();
        void Backward();

        // Getter Function
        int getFiltergetHeight();
        int getFiltergetWidth();
    };

}

#endif
