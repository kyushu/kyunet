#ifndef _SIGMOID_LAYER_H_
#define _SIGMOID_LAYER_H_

#include "layer/layer.h"
#include "activator/sigmoid_act.h"

namespace mkt {

    template<typename T>
    class SigmoidLayer: public Layer<T>
    {
    private:
        Sigmoid_Act<T> sigmoid_act_;
    public:
        SigmoidLayer(Layer<T>* prevLayer, std::string id);
        ~SigmoidLayer();

        void initialize(NetMode mode);
        // Computation Function
        void Forward();
        void Backward();
    };
}
#endif
