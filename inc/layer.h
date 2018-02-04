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

        int batchSize_;

        // result tensor
        int oh_; // DstTensor height
        int ow_; // DstTensor widht
        int oc_; // DstTensor depth (channel)

        // kernel(filter) tensor
        int fh_; // filter height
        int fw_; // filter width
        int fc_; // filter channel = number of Filter(kernel)




    public:
        Layer(
            LayerType type,
            ActivationType activationType = ActivationType::NONE,
            InitializerType weightInitType = InitializerType::NONE,
            InitializerType biasInitType = InitializerType::ZERO
        ):
            type_{type},
            activationType_{activationType},
            weightInitType_{weightInitType},
            biasInitType_{biasInitType},
            batchSize_{0},
            oh_{0}, ow_{0}, oc_{0},
            fh_{0}, fw_{0}, fc_{0},
            pSrc_{nullptr},
            pDst_{nullptr},
            pW_{nullptr},
            pB_{nullptr}
        {};

        virtual ~Layer() {
            fprintf(stderr, "--------------------- Layer Destructor\n");

            pSrc_ = nullptr;
            fprintf(stderr, "--------------------- Layer Destructor pSrc_\n");

            fprintf(stderr, "pDst_.adr: %p\n", pDst_);
            fprintf(stderr, "pDst_->pData.adr: %p\n", pDst_->pData_);
            delete pDst_;
            fprintf(stderr, "--------------------- Layer Destructor pDst_\n");

            fprintf(stderr, "pW_.adr: %p\n", pW_);
            fprintf(stderr, "pW_->pData.adr: %p\n", pW_->pData_);
            delete pW_;
            fprintf(stderr, "--------------------- Layer Destructor pW_\n");

            fprintf(stderr, "pB_.adr: %p\n", pB_);
            fprintf(stderr, "pB_->pData.adr: %p\n", pB_->pData_);
            delete pB_;
            fprintf(stderr, "--------------------- Layer Destructor pB_\n");

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
