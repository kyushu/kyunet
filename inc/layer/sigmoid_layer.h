#ifndef _SIGMOID_LAYER_H_
#define _SIGMOID_LAYER_H_

#include "layer/layer.h"
#include "activator/sigmoid_act.h"

namespace mkt {

    class SigmoidLayer: public Layer
    {
    private:
        Sigmoid_Act sigmoid_act_;
    public:
        SigmoidLayer(Layer* prevLayer, std::string id);
        ~SigmoidLayer();

        void initialize();
        // Computation Function
        void Forward();
        void Backward();
    };
}
#endif
