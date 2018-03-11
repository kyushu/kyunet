#ifndef MKT_RELU_H
#define MKT_RELU_H

#include "activator/activator.h"


namespace mkt {

    class Relu_Act: public Activator
    {
    private:
        float negative_slope_;
    public:
        Relu_Act();
        ~Relu_Act();

        void Forward(Tensor &src, Tensor &dst);
        void Backward(Tensor &src, Tensor &src_grad, Tensor &dst_grad);

    };
}



#endif
