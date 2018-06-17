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


    struct ConvParam
    {
        PaddingType paddingType_;
        int stride_w_;
        int stride_h_;
        int pad_w_;
        int pad_h_;
        int dilation_w_;
        int dilation_h_;

        // Constructor without parameter
        ConvParam():paddingType_{PaddingType::VALID}, stride_w_{1}, stride_h_{1}, pad_w_{0}, pad_h_{0}, dilation_w_{1}, dilation_h_{1} {}

        // Constructor with parameters
        explicit ConvParam(PaddingType paddingType, int stride_w, int stride_h, int pad_w, int pad_h, int dilation_w=1, int dilation_h=1): paddingType_{paddingType}, stride_w_{stride_w}, stride_h_{stride_h}, pad_w_{pad_w}, pad_h_{pad_h}, dilation_w_{dilation_w}, dilation_h_{dilation_h} {}

        // Copy constructor
        ConvParam(const ConvParam& convParam) {
            paddingType_ = convParam.paddingType_;
            stride_w_    = convParam.stride_w_;
            stride_h_    = convParam.stride_h_;
            pad_w_       = convParam.pad_w_;
            pad_h_       = convParam.pad_h_;
            dilation_w_  = convParam.dilation_w_;
            dilation_h_  = convParam.dilation_h_;
        }

    };

    struct Shape
    {
        int num_;
        int depth_;
        int height_;
        int width_;
        Shape():num_{0}, depth_{0}, height_{0}, width_{0} {}
        explicit Shape(int num, int depth, int height, int width): num_{num}, depth_{depth}, height_{height}, width_{width}{}

        int operator [] (int i) {
            switch(i) {
                case 0:
                    return num_;
                    break;
                case 1:
                    return depth_;
                    break;
                case 2:
                    return height_;
                    break;
                case 3:
                    return width_;
                    break;
                default:
                    return 0;
            }
        }

        int getNum()    { return num_;    }
        int getDepth()  { return depth_;  }
        int getHeight() { return height_; }
        int getWidth()  { return width_;  }
    };

} // namespace mkt




#endif
