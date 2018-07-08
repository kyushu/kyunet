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
#include <fstream>

#include "stb_image.h"

#include "solver/solver.h"

#include "layer/layer.h"
#include "layer/input_layer.h"
#include "layer/dense_layer.h"
#include "layer/conv_layer.h"
#include "layer/relu_layer.h"
#include "layer/sigmoid_layer.h"
#include "layer/pooling_layer.h"
#include "layer/softmax_layer.h"
#include "layer/cross_entropy_loss_with_softmax_layer.h"
#include "layer/batchNorm_layer.h"



namespace mkt {

    template<typename T>
    class Solver;

    template<typename T>
    class KyuNet
    {
    private:



    public:
        std::vector<Layer<T>* > layers_;
        InputLayer<T>* pInputLayer_;
        Solver<T>* pSolver_;
        //==================================
        KyuNet();
        ~KyuNet();
        //==================================

        // Add Solver Function
        void addSolver(Solver<T>* pSolver);

        // Add Layer Function
        Layer<T>* addInputLayer(std::string id, int batchSize, int h, int w, int c);

        Layer<T>* addDenseLayer(Layer<T>* prevLayer, std::string id, int unit, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType);

        Layer<T>* addDenseLayer(Layer<T>* prevLayer, std::string id, LayerParams params);

        // Layer<T>* addConvLayer(Layer<T>* prevLayer, std::string id, int kernel_Height, int kernel_width, int kernel_channel, int stride_h, int stride_w, int pad_h, int pad_w, PaddingType paddingType, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType);

        Layer<T>* addConvLayer(Layer<T>* prevLayer, std::string id, int kernel_Height, int kernel_width, int kernel_channel, ConvParam convParam, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType);

        Layer<T>* addConvLayer(Layer<T>* prevLayer, std::string id, LayerParams params);

        Layer<T>* addReluLayer(Layer<T>* prevLayer, std::string id);

        Layer<T>* addSigmoidLayer(Layer<T>* prevLayer, std::string id);

        Layer<T>* addPoolingLayer( Layer<T>* prevLayer, std::string id, int kernel_Height, int kernel_width, ConvParam convParam, PoolingMethodType type);

        Layer<T>* addPoolingLayer( Layer<T>* prevLayer, std::string id, LayerParams params);

        Layer<T>* addSoftmaxLayer( Layer<T>* prevLayer, std::string id);

        Layer<T>* addCrossEntropyLossWithSoftmaxLayer( Layer<T>* prevLayer, std::string id);

        Layer<T>* addBatchNormLayer(Layer<T>* prevLayer, std::string id);
        /**
         * Compile Function: allocate memory
         */
        void Compile(NetMode mode);
        // Forward Function
        void Forward();

        // Backward Function
        void Backward();

        void Train();

        OP_STATUS SaveModel(std::string file_path, bool bWriteInfo);
        OP_STATUS LoadModel(std::string file_path, bool bWriteInfo);

        // Add Data Function
        OP_STATUS add_data_from_file_list(std::vector<std::string> fileList);

        void addBatchLabels(std::string layer, std::vector<int> labels);
        // Helper Function
        void deFlattenInputImage(unsigned char *pImg, int index, int max_pixel_value=255);
        // Getter
        int getNumOfLayer();
        const InputLayer<T>* getInputLayer();
        const std::vector<Layer<T>*> getLayers();

    };

}

#endif
