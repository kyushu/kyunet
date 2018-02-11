#ifndef _SIGMOID_H_
#define _SIGMOID_H_

#include <math.h>
#include "activator/activator.h"


namespace mkt {




    class Sigmoid_Act: public Activator
    {
    private:
        float negative_slope_;
    public:
        Sigmoid_Act();
        ~Sigmoid_Act();

        void forward(Tensor &src, Tensor &dst);
        void backward(Tensor &src, Tensor &src_grad, Tensor &dst_grad);

    };
}



#endif
