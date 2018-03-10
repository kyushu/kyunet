#ifndef MKT_LOSS_LAYER_H
#define MKT_LOSS_LAYER_H

#include "layer.h"

namespace mkt {
    class LossLayer: public Layer
    {
    public:
        LossLayer(Layer* prevLayer, std::string id, int numClass);
        ~LossLayer();

        void initialize();
    };
}


#endif
