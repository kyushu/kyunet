#include "layer/loss_layer.h"

namespace mkt {

    // Constructor
    LossLayer::LossLayer(Layer* prevLayer, std::string id, int numClass): Layer(LayerType::Softmax) {

        batchSize_ = prevLayer->pDst_->NumOfData();
        pSrc_ = prevLayer->pDst_;

        int ih = prevLayer->pDst_->Height();
        int iw = prevLayer->pDst_->Width();
        int ic = prevLayer->pDst_->Channel();

        oh_ = 1;
        ow_ = 1;
        oc_ = 1;

        // pDst_ = loss
        pDst_ = new Tensor{1, oh_, ow_, oc_};
    }

    // Destructor
    LossLayer::~LossLayer() {
    }

    /*
     * Initialize
     */
    void LossLayer::initialize() {
        initOutputTensor();
    };
}
