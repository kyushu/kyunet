#ifndef _DENSE_LAYER_H_
#define _DENSE_LAYER_H_

#include "layer.h"

namespace mkt {

    class DenseLayer: public Layer
    {

    public:
        int unit_;

        // Constructor with ID
        DenseLayer(
            Layer* prevLayer,
            std::string id,
            int unit,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        );

        // Constructor without ID
        DenseLayer(
            Layer* prevLayer,
            int unit,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        );

        ~DenseLayer();

        void initialize();

        // Computation Function
        void forward();
        void backward();
    };

}

#endif
