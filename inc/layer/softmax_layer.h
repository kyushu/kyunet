#ifndef MKT_SOFTMAX_LAYER_H
#define MKT_SOFTMAX_LAYER_H

#include "layer/layer.h"
#include "operators/mat_operators.h"

namespace mkt {
    class SoftmaxLayer: public Layer
    {
    public:
        Tensor* pScale_;

        SoftmaxLayer();
        SoftmaxLayer(Layer* prevLayer, std::string id);
        ~SoftmaxLayer();

        void initialize();
        void Reshape(int num, int height, int width, int ch);

        // Computation Function
        void Forward();
        void Backward();
    };
}



#endif
