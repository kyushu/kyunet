#ifndef _CONVLAYER_H_
#define _CONVLAYER_H_

#include "layer.h"

namespace mkt {
    class ConvLayer
    {
    public:
        ConvLayer(
            Layer* prevLayer,
            std::string id_,
            int filters_, int kernelSize_, int stride_, int padding_, ,
            ActivationType actType_,
            InitializerType initType_):
        Layer(LayerType::FullConnected, actType_, initType_);
        ~ConvLayer();

    };

}


#endif
