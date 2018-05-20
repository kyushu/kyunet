#ifndef MKT_SOFTMAX_LAYER_H
#define MKT_SOFTMAX_LAYER_H

#include "layer/layer.h"
#include "operators/mat_operators.h"

namespace mkt {

    template<typename T>
    class SoftmaxLayer: public Layer<T>
    {
    public:
        Tensor<T>* pScale_;

        SoftmaxLayer();
        SoftmaxLayer(Layer<T>* prevLayer, std::string id);
        ~SoftmaxLayer();

        void initialize(NetMode mode);
        void Reshape(int num, int height, int width, int ch);

        // Computation Function
        void Forward();
        void Backward();
    };
}



#endif
