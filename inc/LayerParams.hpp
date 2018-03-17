#ifndef PARAMETER_HPP
#define PARAMETER_HPP

namespace mkt {

    struct LayerParams
    {
        int kernel_channel;
        int kernel_height;
        int kernel_width;

        int stride_h;
        int stride_w;

        int pad_h;
        int pad_w;
        PaddingType paddingType;

        int dilation_h;
        int dilation_w;

        ActivationType actType;

        InitializerType weight_init_type;
        InitializerType bias_init_type;

        // constructor
        LayerParams():
            kernel_channel{0},
            kernel_height{0},
            kernel_width{0},
            stride_h{0},
            stride_w{0},
            pad_h{0},
            pad_w{0},
            paddingType{PaddingType::VALID},
            dilation_h{1},
            dilation_w{1},
            actType{ActivationType::NONE},
            weight_init_type{InitializerType::NONE},
            bias_init_type{InitializerType::NONE}
        {};

        LayerParams(
            int kh,
            int kw,
            int kc,
            int sh,
            int sw,
            int ph,
            int pw,
            PaddingType pType,
            ActivationType aType,
            InitializerType wInitType,
            InitializerType bInitType
        ):
            kernel_channel{kc},
            kernel_height{kh},
            kernel_width{kw},
            stride_h{sh},
            stride_w{sw},
            pad_h{ph},
            pad_w{pw},
            paddingType{pType},
            weight_init_type{wInitType},
            bias_init_type{bInitType}
        {};
    };
}




#endif
