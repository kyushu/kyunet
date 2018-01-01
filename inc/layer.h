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

#include "tensor.h"

namespace mkt {

    template<class DType>
    class Layer
    {
    protected:
        LayerType type_;

    public:
        Tensor<DType> *pSrcTensor_;      // point to dst_tensor of previous layer
        Tensor<DType> *pDstTensor_;      // new a destination tensor for self use
        Tensor<DType> *pWTensor_;   // new a weight tensor for self use
        DType* pBias_;          // new a chunk of memory for self use
        int dh_; // DstTensor height
        int dw_; // DstTensor widht
        int dc_; // DstTensor depth (channel)
        
        int fh_; // filter height
        int fw_; // filter width
        int fc_; // filter channel


    public:
        Layer(LayerType type):
            type_{type}, 
            dh_{0}, dw_{0}, dc_{0},
            fh_{0}, fw_{0}, fc_{0},
            pSrcTensor_{nullptr}, pDstTensor_{nullptr}, pWTensor_{nullptr}, pBias_{nullptr}
        {};

        virtual ~Layer()
        {
            pSrcTensor_ = nullptr;
            delete pDstTensor_;
            delete pWTensor_;
            delete[] pBias_;
        };

        // TODO: copy constructor

        // Getter
        LayerType getType();

        void forward();     // forward pass
        void backward();    // back propagation



        static int gemm_nr(int trans_a, int trans_b, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc);
        
    };
}


#endif