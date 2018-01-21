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

#ifndef _LAYER_H_
#define _LAYER_H_

#include <iostream>
#include <string>

#include "operation.h"
#include "tensor.h"

namespace mkt {


    class Layer
    {
    protected:
        LayerType type;
        InitializerType weightInitType;
        InitializerType biasInitType;
        ActivationType activationType;

    public:
        std::string id;
        Tensor *pSrc;    // point to dst_tensor of previous layer
        Tensor *pDst;    // new a destination tensor
        Tensor *pW;      // new a weight tensor
        Tensor* pB;      // new a bias tensor

        int batchSize;

        int dh; // DstTensor height
        int dw; // DstTensor widht
        int dc; // DstTensor depth (channel)

        int fh; // filter height
        int fw; // filter width
        int fc; // filter channel = number of Filter(kernel)


    public:
        Layer(
            LayerType type,
            ActivationType activationType_ = ActivationType::NONE,
            InitializerType weightInitType_ = InitializerType::NONE,
            InitializerType biasInitType_ = InitializerType::ZERO
        ):
            type{type},
            activationType{activationType_},
            weightInitType{weightInitType_},
            biasInitType{biasInitType_},
            batchSize{0},
            dh{0}, dw{0}, dc{0},
            fh{0}, fw{0}, fc{0},
            pSrc{nullptr},
            pDst{nullptr},
            pW{nullptr},
            pB{nullptr}
        {};

        virtual ~Layer() {
            pSrc = nullptr;
            delete pDst;
            delete pW;
            delete pB;
        };

        // TODO: copy constructor

        // Initialize Function
        virtual void initialize()=0;
        void initOutputTensor();
        void initWeightTensor(InitializerType initType);
        void initBiasTensor(InitializerType initType);

        // Computation Function
        virtual void forward()=0;     // forward pass
        virtual void backward()=0;    // back propagation
        void addBias();
        void applyActivation();

        // Getter function
        LayerType getType();
        InitializerType getWeightInitType();
        InitializerType getBiasInitType();
        ActivationType getActivationType();
        int getBatchSize();
        int getOutputHeight();
        int getOutputWidth();
        int getOutputChannel();
        int getFilterHeight();
        int getFilterWidth();
        int getFilterChannel();




        // Data

    };
}


#endif
