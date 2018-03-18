#ifndef MKT_DENSE_LAYER_H
#define MKT_DENSE_LAYER_H

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
        DenseLayer(Layer* prevLayer, std::string id, LayerParams params);

        ~DenseLayer();

        void initialize();

        // Computation Function
        void Forward();
        void Backward();
    };

}

#endif
