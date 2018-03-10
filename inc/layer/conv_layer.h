#ifndef MKT_CONV_LAYER_H
#define MKT_CONV_LAYER_H

#include "layer.h"
#include <math.h>

namespace mkt {
    class ConvLayer:public Layer
    {
    public:
        int fh_; // filter height
        int fw_; // filter width
        int fc_; // filter channel = number of Filter(kernel)
        int stride_h_;
        int stride_w_;
        int pad_h_;
        int pad_w_;
        PaddingType paddingType_;

        // kernel(filter) tensor Dimension
        int dilation_h_;
        int dilation_w_;

        Tensor* pTmpCol_;


        ConvLayer(
            Layer* prevLayer,
            std::string id,
            int kernel_Height,
            int kernel_width,
            int kernel_channel,
            int stride_h,
            int stride_w,
            int pad_h,
            int pad_w,
            PaddingType paddingType,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        );

        ~ConvLayer();

        void initialize();

        // Computation Function
        void Forward();
        void Backward();

        // Getter Function
        int getFilterHeight();
        int getFilterWidth();
        int getFilterChannel();
    };

}


#endif
