#ifndef MKT_CONV_LAYER_H
#define MKT_CONV_LAYER_H

#include <math.h>
#include "layer.h"



namespace mkt {
    template<typename T>
    class ConvLayer:public Layer<T>
    {
    public:

        ConvLayer(
            Layer<T>* prevLayer,
            std::string id,
            int fh,
            int fw,
            int fc,
            // int stride_h,
            // int stride_w,
            // int pad_h,
            // int pad_w,
            // PaddingType paddingType,
            ConvParam convParam,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        );

        ConvLayer(Layer<T>* prevLayer, std::string id, LayerParams params);

        ~ConvLayer();

        // Must Implement virtual finctions form Layer class
        void initialize(NetMode mode);
        void Forward();
        void Backward();

        void InferShape ();
        // Getter Function
        int getFiltergetHeight();
        int getFiltergetWidth();
        int getFiltergetChannel();
        Tensor<T>* getTmpCol();

    private:
        int fh_; // filter height
        int fw_; // filter width
        int fc_; // filter channel = number of Filter(kernel)

        // int stride_h_;
        // int stride_w_;
        // int pad_h_;
        // int pad_w_;
        // PaddingType padding_type_;
        // int dilation_h_;
        // int dilation_w_;
        ConvParam convParam_;
        Tensor<T>* pTmpCol_;
    };
} // namespace mkt



#endif
