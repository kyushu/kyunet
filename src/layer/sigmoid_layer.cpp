#include "layer/sigmoid_layer.h"

namespace mkt {

    SigmoidLayer::SigmoidLayer(
        Layer* prevLayer,
        std::string id
    ): Layer(LayerType::Relu)
    {

        batchSize_ = prevLayer->pDst_->getNumOfData();
        pSrc_ = prevLayer->pDst_;

        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getDepth();

        oh_ = ih;
        ow_ = iw;
        oc_ = ic;

        pDst_ = new Tensor{batchSize_, ih, iw, ic};
    };

    SigmoidLayer::~SigmoidLayer() {};

    void SigmoidLayer::initialize() {
        initOutputTensor();
    };


    void SigmoidLayer::forward() {
        sigmoid_act_.forward(*pSrc_, *pDst_);
    };

    void SigmoidLayer::backward() {

    };
}
