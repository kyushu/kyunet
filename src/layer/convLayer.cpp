#include "convLayer.h"

namespace mkt {

    ConvLayer::ConvLayer(
            Layer* prevLayer,
            std::string id,
            int nfilter, int kernelSize, int stride, int padding,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType,
            PaddingType paddingType
    ):
        nfilter_{nfilter},
        kernelSize_{kernelSize},
        stride_{stride},
        padding_{padding},
        paddingType_{paddingType}
        Layer(LayerType::Convolution, actType, weightInitType, biasInitType)
    {

        int batchSize = prevLayer->pDst_->getNumOfData()();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getDepth();
        int size3D = prevLayer->pDst_->getSize3D();

        oc_ = nfilter;
        // calculate pDst size: W_o = (W-f +2p)/s +1
        switch(paddingType_) {
            case valid:
                ow_ = std::floor(float(iw - kernelSize_) / stride_) + 1;
                oh_ = std::floor(float(ih - kernelSize_) / stride_) + 1;
                break;
            case same:
                padding_ = ((iw-1)*stride_ + kernelSize_ - iw)*0.5;
                ow_ = int( (iw - kernelSize_ + (2*padding_)) / stride_ ) + 1;
                oh_ = int( (ih - kernelSize_ + (2*padding_)) / stride_ ) + 1;
                break;
            default:
                ow_ = std::floor(float(iw - kernelSize_) / stride_) + 1;
                oh_ = std::floor(float(ih - kernelSize_) / stride_) + 1;
                break;
        }

        pDst_ = new Tensor{batchSize, oh, ow, oc};
        pW_   = new Tensor{ic, kernelSize_, kernelSize_, oc};
        pB_   = new Tensor{1, 1, 1, oc};

        // TODO: Activation
    }


    ConvLayer::~ConvLayer() {

    }

    // Computation Function
    void ConvLayer::forward() {

    }
    void ConvLayer::backward() {

    }
}
