/*
 * Cross_entropy_loss_with_softmax_layer is a wrapper of SoftmaxLayer and cross entropy loss
 */

#ifndef _MKT_CROSS_ENTROPY_LOSS_WITH_SOFTMAX_LAYER
#define _MKT_CROSS_ENTROPY_LOSS_WITH_SOFTMAX_LAYER

#include "layer.h"
#include "softmax_layer.h"

namespace mkt {

    class CrossEntropyLossWithSoftmaxLayer: public Layer
    {
    public:
        CrossEntropyLossWithSoftmaxLayer(Layer* prevLayer, std::string id);
        ~CrossEntropyLossWithSoftmaxLayer();

        void initialize();

        void LoadLabel(const int num, const int* label);
        void LoadLabel(const std::vector<int>& labels);

        void Forward();
        void Backward();

        SoftmaxLayer softmaxLayer_;
        Tensor* pLabel_;
    };
}



#endif
