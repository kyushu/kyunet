#ifndef MKT_ACTIVATION_H
#define MKT_ACTIVATION_H

#include "tensor.h"
#include "help_func/helper.hpp"
#include "mkt_log.h"

namespace mkt {

    // Abstract Class
    template<typename T>
    class Activator
    {
    public:
        virtual ~Activator(){};
        virtual void Forward(Tensor<T> *src, Tensor<T> *dst)=0;                           // forward pass
        virtual void Backward(Tensor<T> *src, Tensor<T> *src_grad, Tensor<T> *dst_grad)=0;   // back propagation
    };


}

#endif
