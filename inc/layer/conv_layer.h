#ifndef _CONV_LAYER_H_
#define _CONV_LAYER_H_

#include "layer.h"
#include <math.h>

namespace mkt {
    class ConvLayer:public Layer
    {
    public:
        int nfilter_;
        int kernelSize_;
        int stride_;
        int padding_;
        PaddingType paddingType_;

        int dilation_h_;
        int dilation_w_;

        Tensor* pTmpCol_;


        ConvLayer(
            Layer* prevLayer,
            std::string id,
            int nfilter, int kernelSize, int stride,
            int padding, PaddingType paddingType,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        );

        ~ConvLayer();

        void initialize();

        // Computation Function
        void forward();
        void backward();

    };

}


#endif
