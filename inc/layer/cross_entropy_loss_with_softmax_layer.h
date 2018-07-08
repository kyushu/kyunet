/*
 * Cross_entropy_loss_with_softmax_layer is a wrapper of SoftmaxLayer and cross entropy loss function
 */

#ifndef MKT_CROSS_ENTROPY_LOSS_WITH_SOFTMAX_LAYER_H
#define MKT_CROSS_ENTROPY_LOSS_WITH_SOFTMAX_LAYER_H

#include "layer.h"
#include "softmax_layer.h"

#include "operations/aux_operations.h"

namespace mkt {
    template<typename T>
    class CrossEntropyLossWithSoftmaxLayer: public Layer<T>
    {
    public:
        CrossEntropyLossWithSoftmaxLayer(Layer<T>* prevLayer, std::string id);
        ~CrossEntropyLossWithSoftmaxLayer();

        // Must Implement virtual finctions form Layer class
        void initialize(NetMode mode);
        void Forward();
        void Backward();
        void serialize(std::fstream& fileHandler, bool bWriteInfo);
        void deserialize(std::fstream& fileHandler, bool bWriteInfo);


        void LoadLabel(const int num, const int* label);
        void LoadLabel(const std::vector<int>& labels);



        SoftmaxLayer<T> softmaxLayer_;
        Tensor<T>* pLabel_;
    };
}



#endif
