#include "layer/sigmoid_layer.h"

namespace mkt {

    SigmoidLayer::SigmoidLayer(
        Layer* prevLayer,
        std::string id
    ): Layer(LayerType::Sigmoid)
    {
        id_ = id;

        batchSize_ = prevLayer->pDst_->NumOfData();

        pPrevLayer_ = prevLayer;

        int ih = prevLayer->pDst_->Height();
        int iw = prevLayer->pDst_->Width();
        int ic = prevLayer->pDst_->Channel();

        oh_ = ih;
        ow_ = iw;
        oc_ = ic;

        pDst_  = new Tensor{batchSize_, oh_, ow_, oc_};
        pgDst_ = new Tensor{batchSize_, oh_, ow_, oc_};

    };

    SigmoidLayer::~SigmoidLayer() {};

    void SigmoidLayer::initialize() {
        initOutputTensor();
    };


    void SigmoidLayer::Forward() {

        Tensor* pSrc = pPrevLayer_->pDst_;

        sigmoid_act_.Forward(*pSrc, *pDst_);
    };

    void SigmoidLayer::Backward() {

    };
}
