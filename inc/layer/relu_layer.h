#ifndef _RELU_LAYER_H_
#define _RELU_LAYER_H_

#include "layer/layer.h"
#include "activator/relu_act.h"

namespace mkt {

    class ReluLayer: public Layer
    {
    private:
        Relu_Act relu_act_;
        float negative_slope_;
    public:
        ReluLayer(Layer* prevLayer, std::string id, float negative_slope = 0.0f);
        ~ReluLayer();

        void initialize();
        // Computation Function
        void forward();
        void backward();
    };
}
#endif
