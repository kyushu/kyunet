#ifndef MKT_MATH_OPERATION_H
#define MKT_MATH_OPERATION_H

#include <cstring>

#include "common_inc.h"
#include "definitions.h"

namespace mkt {

    int gemm_cpu(
        CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
        int M, int N, int K,
        float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

    //
    int axpy(int n, float a, float *x, float *y);

    //
    void im2col_cpu(const float* data_im,
        const int channels, const int height, const int width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        float* data_col);

    void col2im_cpu(const float* data_col, const int channels,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        float* data_im);

    int gemv_cpu(
        CBLAS_TRANSPOSE trans_a,
        int m,
        int n,
        float alpha,
        float *a,
        float *x,
        float beta,
        float *y);

    void set_memory(const int N, const float alpha, float* Y);
}


#endif
