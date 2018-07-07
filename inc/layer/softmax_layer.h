#ifndef MKT_SOFTMAX_LAYER_H
#define MKT_SOFTMAX_LAYER_H

#include "layer/layer.h"
#include "operations/aux_operations.h"
#include "operations/mat_operations.h"

namespace mkt {

    template<typename T>
    class SoftmaxLayer: public Layer<T>
    {
    public:
        Tensor<T>* pScale_;

        SoftmaxLayer();
        SoftmaxLayer(Layer<T>* prevLayer, std::string id);
        ~SoftmaxLayer();

        // Must Implement virtual finctions form Layer class
        void initialize(NetMode mode);
        void Forward();
        void Backward();

        void Reshape(int num, int ch, int height, int width);
    };
}



#endif
