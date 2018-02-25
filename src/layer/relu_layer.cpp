#include "layer/relu_layer.h"

namespace mkt {

    ReluLayer::ReluLayer(
        Layer* prevLayer,
        std::string id,
        float negative_slope
    ): negative_slope_{negative_slope}, Layer(LayerType::Relu)
    {

        batchSize_ = prevLayer->pDst_->getNumOfData();
        pSrc_ = prevLayer->pDst_;

        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getDepth();

        oh_ = ih;
        ow_ = iw;
        oc_ = ic;

        pDst_ = new Tensor{batchSize_, oh_, ow_, oc_};
    };

    ReluLayer::~ReluLayer() {};

    void ReluLayer::initialize() {
        initOutputTensor();
    };


    void ReluLayer::forward() {
        relu_act_.forward(*pSrc_, *pDst_);
    };

    void ReluLayer::backward() {

    };
}
