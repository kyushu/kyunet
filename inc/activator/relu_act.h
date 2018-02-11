#ifndef _RELU_H_
#define _RELU_H_

#include "activator/activator.h"


namespace mkt {

    class Relu_Act: public Activator
    {
    private:
        float negative_slope_;
    public:
        Relu_Act();
        ~Relu_Act();

        void forward(Tensor &src, Tensor &dst);
        void backward(Tensor &src, Tensor &src_grad, Tensor &dst_grad);

    };
}



#endif
