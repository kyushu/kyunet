#ifndef _CONVLAYER_H_
#define _CONVLAYER_H_

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


        ConvLayer(
            Layer* prevLayer,
            std::string id,
            int nfilter, int kernelSize, int stride, int padding,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType,
            PaddingType paddingType
        );

        ~ConvLayer();


        // Computation Function
        void forward();
        void backward();

    };

}


#endif
