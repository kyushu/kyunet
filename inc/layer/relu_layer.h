#ifndef MKT_RELU_LAYER_H
#define MKT_RELU_LAYER_H

#include "layer/layer.h"
#include "activator/relu_act.h"

namespace mkt {

    template<typename T>
    class ReluLayer: public Layer<T>
    {
    private:
        Relu_Act<T> relu_act_;
        T negative_slope_;
    public:
        ReluLayer(Layer<T>* prevLayer, std::string id, T negative_slope = 0.0f);
        ~ReluLayer();

        // Must Implement virtual finctions form Layer class
        void initialize(NetMode mode);
        void Forward();
        void Backward();
    };
}
#endif
