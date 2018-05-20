#ifndef MKT_CONV_LAYER_H
#define MKT_CONV_LAYER_H

#include <math.h>
#include "layer.h"


namespace mkt {
    template<typename T>
    class ConvLayer:public Layer<T>
    {
    public:
        int fh_; // filter height
        int fw_; // filter width
        int fc_; // filter channel = number of Filter(kernel)
        int stride_h_;
        int stride_w_;
        int pad_h_;
        int pad_w_;
        PaddingType padding_type_;

        // kernel(filter) tensor Dimension
        int dilation_h_;
        int dilation_w_;

        Tensor<T>* pTmpCol_;


        ConvLayer(
            Layer<T>* prevLayer,
            std::string id,
            int fh,
            int fw,
            int fc,
            int stride_h,
            int stride_w,
            int pad_h,
            int pad_w,
            PaddingType paddingType,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        );

        ConvLayer(Layer<T>* prevLayer, std::string id, LayerParams params);

        ~ConvLayer();

        void initialize(NetMode mode);

        // Computation Function
        void Forward();
        void Backward();

        void calcOutputSize(int ic, int ih, int iw);
        // Getter Function
        int getFiltergetHeight();
        int getFiltergetWidth();
        int getFiltergetChannel();
    };

}


#endif
