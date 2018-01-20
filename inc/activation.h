#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

namespace mkt {

    class Activation
    {
    public:
        ActivationType type_;
        Activation(ActivationType type=ActivationType::None): type_{type} {
        };

        vitural ~Activation(){};

    };

}

#endif
