#include "convLayer.h"

namespace mkt {

    ConvLayer::ConvLayer(
            Layer* prevLayer,
            std::string id,
            int nfilter, int kernelSize, int stride, int padding,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType,
            PaddingType padding):
        nfilter_{nfilter},
        kernelSize_{kernelSize},
        stride_{stride},
        padding_{padding},
        Layer(LayerType::FullConnected, actType, weightInitType, biasInitType)
    {

        // calculate pDst size

    }

}
