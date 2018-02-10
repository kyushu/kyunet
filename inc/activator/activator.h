#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

namespace mkt {

    class Activation
    {
    public:
        virtual ~Activation(){};

        virtual void forward()=0;     // forward pass
        virtual void backward()=0;    // back propagation
    };


}

#endif
