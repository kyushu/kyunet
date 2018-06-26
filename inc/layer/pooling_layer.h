#ifndef MKT_POOL_LAYER_H
#define MKT_POOL_LAYER_H

#include "layer.h"

namespace mkt {

    template<typename T>
    class PoolingLayer: public Layer<T>
    {
    public:
        int fh_; // filter height
        int fw_; // filter width
        // int stride_h_;
        // int stride_w_;
        // int pad_h_;
        // int pad_w_;
        ConvParam convParam_;
        PoolingMethodType type_;

        Tensor<T>* pMask_;


        PoolingLayer(Layer<T>* prevLayer, std::string id,
            int kernel_Height, int kernel_width,
            // int stride_h, int stride_w,
            // int pad_h, int pad_w,
            ConvParam convParam,
            PoolingMethodType type
            );
        PoolingLayer(Layer<T>* prevLayer, std::string id, LayerParams params);

        ~PoolingLayer();

        // Must Implement virtual finctions form Layer class
        void initialize(NetMode mode);
        void Forward();
        void Backward();

        void inferShape();
        // Getter Function
        int getFiltergetHeight();
        int getFiltergetWidth();
    };

}

#endif
