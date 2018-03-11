#ifndef MKT_ACTIVATION_H
#define MKT_ACTIVATION_H

#include "tensor.h"
#include "help_func/helper.hpp"
#include "mkt_log.hpp"

namespace mkt {

    // Abstract Class
    class Activator
    {
    public:
        virtual ~Activator(){};
        virtual void Forward(Tensor &src, Tensor &dst)=0;                           // forward pass
        virtual void Backward(Tensor &src, Tensor &src_grad, Tensor &dst_grad)=0;   // back propagation
    };


}

#endif
