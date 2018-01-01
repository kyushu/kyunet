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

#include "layer.h"

namespace mkt {

    template<class DType>
    LayerType Layer<DType>::getType() {
        return type_;
    }


    /* 
        C = alpha(AxB) + beta*C 
        C{MXN} = A{MXK} * B{KXN}
        M = Number of rows in matrix A.
        N = Number of columns in matrix B.
        K = Number of columns in matrix A and rows in matrix B
    
        lda = Leading dimension of matrix A.
            It cannot be less than K when the order parameter is set to RowMajor, 
            or less than M when the parameter is set to ColumnMajor.
    
        ldb = Leading dimension of matrix B. It cannot be less than N when the order parameter 
            is set to RowMajor, or less than K when it is set to ColumnMajor.
    
        ldc = Leading dimension of matrix C. It cannot be less than N when the order parameter 
            is set to RowMajor, or less than M when it is set to ColumnMajorOrder.
     */
    /*
    
     */

    // bcnn_gemm(0, 0, 
    //     batch_size, dst_size, src_size, 1.0f,
    //     src.data, src_size, 
    //     layer->weight, dst_size, 
    //     1.0f, 
    //     dst.data, dst_size);

    static int gemm_nr(
        int trans_a, int trans_b, 
        int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb, 
        float BETA, 
        float *C, int ldc)
    {
        int i, j, k;

        if (BETA != 1.0f) {
            for (i = 0; i < M; ++i){
                for (j = 0; j < N; ++j){
                    C[i * ldc + j] *= BETA;
                }
            }
        }
        if (!trans_a && !trans_b) {
            for (i = 0; i < M; ++i){
                for (k = 0; k < K; ++k){
                    float tmp = ALPHA * A[i * lda + k];
                    for (j = 0; j < N; ++j) {
                        C[i * ldc + j] += tmp * B[k * ldb + j];
                    }
                }
            }
        }
        else if (trans_a && !trans_b) {
            for (i = 0; i < M; ++i){
                for (k = 0; k < K; ++k){
                    float tmp = ALPHA * A[k * lda + i];
                    for (j = 0; j < N; ++j) {
                        C[i * ldc + j] += tmp * B[k * ldb + j];
                    }
                }
            }
        }
        else if (!trans_a && trans_b) {
            float sum = 0;
            for (i = 0; i < M; ++i){
                for (j = 0; j < N; ++j){
                    sum = 0;
                    for (k = 0; k < K; ++k){
                        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
                    }
                    C[i * ldc + j] += sum;
                }
            }
        }
        else {
            for (i = 0; i < M; ++i){
                for (k = 0; k < K; ++k){
                    float tmp = ALPHA * A[i * lda + k];
                    for (j = 0; j < N; ++j) {
                        C[i * ldc + j] += tmp * B[k * ldb + j];
                    }
                }
            }
        }
        return 0;
    }




    template class Layer<float>;
}