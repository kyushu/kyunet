#ifndef _RELU_H_
#define _RELU_H_

#include "activator/activator.h"
#include "tensor.h"

namespace mkt {

    class Relu: Activation
    {
    private:

    public:
        Relu();
        ~Relu();

        void forward();
        void backward();

    };
}



#endif
