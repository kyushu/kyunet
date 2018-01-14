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
#include "tensor.h"

namespace mkt {


    class Layer
    {
    protected:
        LayerType type_;

    public:
        std::string id_;
        Tensor *pSrc_;    // point to dst_tensor of previous layer
        Tensor *pDst_;    // new a destination tensor for self use
        Tensor *pW_;            // new a weight tensor for self use
        Tensor* pB_;            // new a chunk of memory for self use

        int batchSize;

        int dh_; // DstTensor height
        int dw_; // DstTensor widht
        int dc_; // DstTensor depth (channel)

        int fh_; // filter height
        int fw_; // filter width
        int fc_; // filter channel = number of Filter(kernel)


    public:
        Layer(LayerType type):
            type_{type},
            batchSize{0},
            dh_{0}, dw_{0}, dc_{0},
            fh_{0}, fw_{0}, fc_{0},
            pSrc_{nullptr},
            pDst_{nullptr},
            pW_{nullptr},
            pB_{nullptr}
        {};

        virtual ~Layer() {
            pSrc_ = nullptr;
            delete pDst_;
            delete pW_;
            delete pB_;
        };

        // TODO: copy constructor

        // Initialize Function
        virtual void initialize()=0;
        void initOutputTensor();
        void initWeightTensor();
        void initBiasTenosr();

        // Computation Function
        void forward();     // forward pass
        void backward();    // back propagation

        // Getter function
        LayerType getType();


        // Data

    };
}


#endif
