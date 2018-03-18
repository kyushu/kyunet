/*
* Copyright (c) 2017 Morpheus Tsai.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef MKT_NET_H
#define MKT_NET_H

// #include <cstdio>

#include "stb_image.h"

#include "layer/layer.h"
#include "layer/input_layer.h"
#include "layer/dense_layer.h"
#include "layer/conv_layer.h"
#include "layer/relu_layer.h"
#include "layer/sigmoid_layer.h"
#include "layer/pooling_layer.h"
#include "layer/softmax_layer.h"
#include "layer/cross_entropy_loss_with_softmax_layer.h"


namespace mkt {

    class Net
    {
    private:
        std::vector<Layer* > layers_;
        InputLayer* pInputLayer_;

    public:
        //==================================
        Net();
        ~Net();
        //==================================

        // Configuration Function
        Layer* addInputLayer(std::string id, int batchSize, int h, int w, int c);

        Layer* addDenseLayer(Layer* prevLayer, std::string id, int unit, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType);

        Layer* addConvLayer(Layer* prevLayer, std::string id, int kernel_Height, int kernel_width, int kernel_channel, int stride_h, int stride_w, int pad_h, int pad_w, PaddingType paddingType, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType);
        Layer* addConvLayer(Layer* prevLayer, std::string id, LayerParams params);

        Layer* addReluLayer(Layer* prevLayer, std::string id);

        Layer* addSigmoidLayer(Layer* prevLayer, std::string id);

        Layer* addPoolingLayer( Layer* prevLayer, std::string id, int kernel_Height, int kernel_width, int stride_h, int stride_w, int pad_h, int pad_w, PoolingMethodType type);
        Layer* addPoolingLayer( Layer* prevLayer, std::string id, LayerParams params);

        Layer* addSoftmaxLayer( Layer* prevLayer, std::string id);

        Layer* addCrossEntropyLossWithSoftmaxLayer( Layer* prevLayer, std::string id);

        // Initialize Function
        void initialize();

        // Forward Function
        void Forward();

        // Backward Function
        void Backward();

        // Add Data Function
        OP_STATUS add_data_from_file_list(std::vector<std::string> fileList);

        // Getter
        InputLayer* getInputLayer();

    };

}

#endif
