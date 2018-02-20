#ifndef MKT_SIGMOID_H
#define MKT_SIGMOID_H

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
