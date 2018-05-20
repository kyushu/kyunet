#ifndef MKT_DENSE_LAYER_H
#define MKT_DENSE_LAYER_H

#include "layer.h"

namespace mkt {

    template<typename T>
    class DenseLayer: public Layer<T>
    {

    public:
        int unit_;

        // Constructor with ID
        DenseLayer(
            Layer<T>* prevLayer,
            std::string id,
            int unit,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        );

        // Constructor without ID
        DenseLayer(
            Layer<T>* prevLayer,
            int unit,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        );
        DenseLayer(Layer<T>* prevLayer, std::string id, LayerParams params);

        ~DenseLayer();

        void initialize(NetMode mode);

        // Computation Function
        void Forward();
        void Backward();
    };

}

#endif
