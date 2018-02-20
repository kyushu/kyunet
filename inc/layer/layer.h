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

#ifndef MKT_LAYER_H
#define MKT_LAYER_H

#include <iostream>
#include <string>

#include "operators/mat_operators.h"
#include "tensor.h"
#include "filler.hpp"
#include "activator/activator.h"
#include "activator/relu_act.h"
#include "activator/sigmoid_act.h"


namespace mkt {


    class Layer
    {
    protected:
        LayerType type_;
        InitializerType weightInitType_;
        InitializerType biasInitType_;
        ActivationType activationType_;


    public:
        std::string id_;
        Tensor *pSrc_;    // point to dst_tensor of previous layer
        Tensor *pDst_;    // new a destination tensor
        Tensor *pW_;      // new a weight tensor
        Tensor* pB_;      // new a bias tensor
        Activator* pActivator_;

        int batchSize_;

        // result tensor Dimension
        int oh_; // Dst Tensor height
        int ow_; // Dst Tensor widht
        int oc_; // Dst Tensor depth (channel)




    public:
        Layer(
            LayerType type,
            ActivationType activationType  = ActivationType::NONE,
            InitializerType weightInitType = InitializerType::NONE,
            InitializerType biasInitType   = InitializerType::NONE
        );

        virtual ~Layer();

        // TODO: copy constructor

        // Initialize Function
        virtual void initialize()=0;
        void initOutputTensor();
        void initWeightTensor();
        void initBiasTensor();

        // Computation Function
        virtual void forward()=0;     // forward pass
        virtual void backward()=0;    // back propagation
        void addBias();
        void applyActivator();

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
