#include "layer/relu_layer.h"

namespace mkt {

    ReluLayer::ReluLayer(
        Layer* prevLayer,
        std::string id,
        float negative_slope
    ): negative_slope_{negative_slope}, Layer(LayerType::RELU)
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

    ReluLayer::~ReluLayer() {};

    void ReluLayer::initialize() {

        MKT_Assert(pDst_ != nullptr, "pDst_ is null");
        MKT_Assert(pgDst_ != nullptr, "pgDst_ is null");

        initOutputTensor();
        initGradTensor();
    };


    void ReluLayer::Forward() {
        Tensor* pSrc = pPrevLayer_->pDst_;

        relu_act_.Forward(*pSrc, *pDst_);
    };

    void ReluLayer::Backward() {

    };
}
