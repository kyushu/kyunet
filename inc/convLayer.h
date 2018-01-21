#ifndef _CONVLAYER_H_
#define _CONVLAYER_H_

#include "layer.h"

namespace mkt {
    class ConvLayer
    {
    public:
        int nfilter;
        int kernelSize;
        int stride;
        PaddingType padding;

        ConvLayer(
            Layer* prevLayer,
            std::string id_,
            int nfilter_, int kernelSize_, int stride_, int padding_, ,
            ActivationType actType_,
            InitializerType weightInitType_,
            InitializerType biasInitType_,
            PaddingType padding_
        );

        ~ConvLayer();

    };

}


#endif
