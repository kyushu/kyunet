#ifndef _OPERATION_H_
#define _OPERATION_H_

#include "common_inc.h"

namespace mkt {

    int gemm_cpu(
        int trans_a, int trans_b,
        int M, int N, int K,
        float ALPHA, float BETA,
        float *A, int lda,
        float *B, int ldb,
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
}


#endif
