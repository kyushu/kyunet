#ifndef _CONVLAYER_H_
#define _CONVLAYER_H_

#include "layer.h"

namespace mkt {
    class ConvLayer
    {
    public:
        int nfilter_;
        int kernelSize_;
        int stride_;
        PaddingType padding_;

        ConvLayer(
            Layer* prevLayer,
            std::string id,
            int nfilter, int kernelSize, int stride, int padding,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType,
            PaddingType padding
        );

        ~ConvLayer();

    };

}


#endif
