#ifndef MKT_DENSE_LAYER_H
#define MKT_DENSE_LAYER_H

#include "layer.h"

namespace mkt {

    template<typename T>
    class DenseLayer: public Layer<T>
    {

    public:

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

        // Must Implement virtual finctions form Layer class
        void initialize(NetMode mode);
        void Forward();
        void Backward();
    };

}

#endif
