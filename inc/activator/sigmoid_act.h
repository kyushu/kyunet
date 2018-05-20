#ifndef MKT_SIGMOID_H
#define MKT_SIGMOID_H

#include <math.h>
#include "activator/activator.h"


namespace mkt {

    template<typename T>
    class Sigmoid_Act: public Activator<T>
    {
    private:
        float negative_slope_;
    public:
        Sigmoid_Act();
        ~Sigmoid_Act();

        void Forward(Tensor<T> *src, Tensor<T> *dst);
        void Backward(Tensor<T> *src, Tensor<T> *src_grad, Tensor<T> *dst_grad);

    };
}



#endif
