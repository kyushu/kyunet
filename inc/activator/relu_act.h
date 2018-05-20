#ifndef MKT_RELU_H
#define MKT_RELU_H

#include "activator/activator.h"


namespace mkt {

    template<typename T>
    class Relu_Act: public Activator<T>
    {
    private:
        float negative_slope_;
    public:
        Relu_Act();
        ~Relu_Act();

        void Forward(Tensor<T> *src, Tensor<T> *dst);
        void Backward(Tensor<T> *src, Tensor<T> *src_grad, Tensor<T> *dst_grad);

    };
}



#endif
