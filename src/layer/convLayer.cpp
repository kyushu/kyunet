#include "convLayer.h"

namespace mkt {

    ConvLayer::ConvLayer(
            Layer* prevLayer,
            std::string id_,
            int nfilter_, int kernelSize_, int stride_, int padding_, ,
            ActivationType actType_,
            InitializerType weightInitType_,
            InitializerType biasInitType_,
            PaddingType padding_):
        nfilter{nfilter_},
        kernelSize{kernelSize_},
        stride{stride_},
        padding{padding_},
        Layer(LayerType::FullConnected, actType_, weightInitType_, biasInitType_)
    {

        // calculate pDst size

    }

}
