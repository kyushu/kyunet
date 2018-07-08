#ifndef MKT_SIGMOID_LAYER_H
#define MKT_SIGMOID_LAYER_H

#include "layer/layer.h"
#include "activator/sigmoid_act.h"

namespace mkt {

    template<typename T>
    class SigmoidLayer: public Layer<T>
    {
    private:
        Sigmoid_Act<T> sigmoid_act_;
    public:
        SigmoidLayer(Layer<T>* prevLayer, std::string id);
        ~SigmoidLayer();

        // Must Implement virtual finctions form Layer class
        void initialize(NetMode mode);
        void Forward();
        void Backward();
        void serialize(std::fstream& fileHandler, bool bWriteInfo);
        void deserialize(std::fstream& fileHandler, bool bWriteInfo);
    };
}
#endif
