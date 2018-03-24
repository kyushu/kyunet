#include "layer/loss_layer.h"

namespace mkt {

    // Constructor
    LossLayer::LossLayer(Layer* prevLayer, std::string id, int numClass): Layer(LayerType::Softmax) {

        batchSize_ = prevLayer->pDst_->getNumOfData();

        pPrevLayer_ = prevLayer;

        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

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
