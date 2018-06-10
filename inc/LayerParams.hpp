#ifndef MKT_PARAMETER_HPP
#define MKT_PARAMETER_HPP

#include "definitions.h"

namespace mkt {

    struct LayerParams
    {
        int fc;
        int fh;
        int fw;

        int stride_h;
        int stride_w;

        int pad_h;
        int pad_w;
        PaddingType padding_type;

        int dilation_h;
        int dilation_w;

        ActivationType actType;

        InitializerType weight_init_type;
        InitializerType bias_init_type;

        PoolingMethodType pooling_type;

        // constructor
        LayerParams():
            fc{0},
            fh{0},
            fw{0},
            stride_h{0},
            stride_w{0},
            pad_h{0},
            pad_w{0},
            padding_type{PaddingType::VALID},
            dilation_h{1},
            dilation_w{1},
            actType{ActivationType::NONE},
            weight_init_type{InitializerType::NONE},
            bias_init_type{InitializerType::NONE},
            pooling_type{PoolingMethodType::MAX}
        {};

        LayerParams(
            int kernel_channel,
            int kernel_height,
            int kernel_width,
            int sh,
            int sw,
            int ph,
            int pw,
            PaddingType padType,
            ActivationType aType,
            InitializerType wInitType,
            InitializerType bInitType,
            PoolingMethodType poolingType
        ):
            fc{kernel_channel},
            fh{kernel_height},
            fw{kernel_width},
            stride_h{sh},
            stride_w{sw},
            pad_h{ph},
            pad_w{pw},
            padding_type{padType},
            weight_init_type{wInitType},
            bias_init_type{bInitType},
            pooling_type{poolingType}
        {};
    };
}




#endif
