#include "layer/sigmoid_layer.h"

namespace mkt {

    SigmoidLayer::SigmoidLayer(
        Layer* prevLayer,
        std::string id
    ): Layer(LayerType::SIGMOID)
    {
        id_ = id;

        batchSize_ = prevLayer->pDst_->getNumOfData();

        pPrevLayer_ = prevLayer;

        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

        oh_ = ih;
        ow_ = iw;
        oc_ = ic;

        pDst_  = new Tensor{batchSize_, oh_, ow_, oc_};
        pgDst_ = new Tensor{batchSize_, oh_, ow_, oc_};

    };

    SigmoidLayer::~SigmoidLayer() {};

    void SigmoidLayer::initialize(NetMode mode) {
        MKT_Assert(pDst_ != nullptr, "pDst_ is null");
        MKT_Assert(pgDst_ != nullptr, "pgDst_ is null");

        initOutputTensor();
        initGradTensor();
    };


    void SigmoidLayer::Forward() {

        Tensor* pSrc = pPrevLayer_->pDst_;

        sigmoid_act_.Forward(*pSrc, *pDst_);
    };

    void SigmoidLayer::Backward() {

    };
}
