#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "tensor.h"
#include "help_func/helper.hpp"
#include "mkt_log.hpp"

namespace mkt {

    // Abstract Class
    class Activation
    {
    public:
        virtual ~Activation(){};
        virtual void forward(Tensor &src, Tensor &dst)=0;                           // forward pass
        virtual void backward(Tensor &src, Tensor &src_grad, Tensor &dst_grad)=0;   // back propagation
    };


}

#endif
