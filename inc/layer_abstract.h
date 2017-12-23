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
    class Layer_abstract
    {
    public:
        Tensor<DType> *pSrc_tensor;      // point to dst_tensor of previous layer
        Tensor<DType> *pDst_tensor;      // new a destination tensor for self use
        Tensor<DType> *pWeight_tensor;   // new a weight tensor for self use
        DType* pBias = nullptr;                    // new a chunk of memory for self use
        int fh; // filter height
        int fw; // filter width
        int ch; // filter channel


    public:
        Layer_abstract():
            pSrc_tensor{nullptr}, pDst_tensor{nullptr}, pWeight_tensor{nullptr}, pBias{nullptr}
        {};
        virtual ~Layer_abstract()
        {
            pSrc_tensor = nullptr;
            delete pDst_tensor;
            delete pWeight_tensor;
            delete[] pBias;
        };

        static int gemm_nr(int trans_a, int trans_b, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc);

        void forward();     // forward pass
        void backward();    // back propagation 
        
    };
}


#endif