#include "layer/relu_layer.h"

namespace mkt {

    ReluLayer::ReluLayer(
        Layer* prevLayer,
        std::string id,
        float negative_slope
    ): negative_slope_{negative_slope}, Layer(LayerType::Relu)
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

    ReluLayer::~ReluLayer() {};

    void ReluLayer::initialize() {
        initOutputTensor();

        initGradOutputTensor();
    };


    void ReluLayer::Forward() {
        Tensor* pSrc = pPrevLayer_->pDst_;

        relu_act_.Forward(*pSrc, *pDst_);
    };

    void ReluLayer::Backward() {

    };
}
