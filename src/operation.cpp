#include "operation.h"
#include <iostream>
namespace mkt {


/*
    General Matrix-Matrix multiplication (GEMM):
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

    General Matrix-Matrix multiplication (Row major)
                       lda = K = h*w*c           ldb = N
                   ________________     _________________
                  |                |   |                 |
                  |                |   |                 |
    M(batch size) |       A        | x |        B        | K
                  |                |   |                 |
                  |________________|   |_________________|
                                                   ||
                                        _________________
                                       |     ldc = N     |
                                       |                 |
                                       |        C        | M
                                       |                 |
                                       |_________________|


*/

    int gemm_cpu(
        int trans_a, int trans_b,
        int M, int N, int K,
        float ALPHA, float BETA,
        float *A, int lda,
        float *B, int ldb,
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
                        mktLog(1, "c[%d] = A[%d](%f)B[%d](%f)\n", i*ldc+j, i * lda + k, tmp, k * ldb + j, B[k * ldb + j]);
                    }
                }
            }
        }
        else if (trans_a && !trans_b) {
            for (i = 0; i < M; ++i){
                for (k = 0; k < K; ++k){
                    float tmp = ALPHA * A[k * lda + i];
                    for (j = 0; j < N; ++j) {
                        mktLog(1, "c[%d] = A[%d]B[%d]\n", i*ldc+j, k * lda + i, i * ldc + j);
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
                        mktLog(1, "c[%d] = A[%d]B[%d]\n", i*ldc+j, i*lda+k, j*ldb+k);
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
                        mktLog(1, "c[%d] = A[%d]B[%d]\n", i*ldc+j, i *lda+k, k*ldb+j);
                        C[i * ldc + j] += tmp * B[k * ldb + j];
                    }
                }
            }
        }
        return 0;
    }

    // Result = aX + Y
    int axpy(int n, float a, float *x, float *y)
    {
        for (int i = 0; i < n; ++i)
        {
            y[i] += a * x[i];
        }
    }

    // TODO

    // Function uses casting from int to unsigned to compare if value of
    // parameter a is greater or equal to zero and lower than value of
    // parameter b. The b parameter is of type signed and is always positive,
    // therefore its value is always lower than 0x800... where casting
    // negative value of a parameter converts it to value higher than 0x800...
    // The casting allows to use one condition instead of two.
    inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }

    void im2col_cpu(const float* data_im,
        const int channels, const int height, const int width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        float* data_col)
    {
        const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = height * width;

        for (int channel = channels; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {

                    int input_row = -pad_h + kernel_row * dilation_h;
                    mktLog(1, "c:%d, kr:%d, kc: %d, input_row: %d\n", channels, kernel_row, kernel_col, input_row);

                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                            for (int output_cols = output_w; output_cols; output_cols--) {
                                *(data_col++) = 0;
                            }
                        }
                        else {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            // mktLog(1, "input_col: %d\n", input_col);
                            for (int output_col = output_w; output_col; output_col--) {
                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(data_col++) = data_im[input_row * width + input_col];

                                    // LOG
                                    mktLog(1, "%d - input_row:%d\n", output_col, input_row);
                                    mktLog(1, "%d - width:%d\n",     output_col, width);
                                    mktLog(1, "%d - input_col:%d\n", output_col, input_col);
                                    mktLog(1, "input_row * width + input_col: %d\n", input_row * width + input_col);
                                    // LOG
                                } else {
                                    *(data_col++) = 0;
                                }
                                input_col += stride_w;
                            }
                        }
                        input_row += stride_h;
                    }
                }
            }
        }
    }


    void col2im_cpu(const float* data_col, const int channels,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        float* data_im)
    {
        // caffe_set(height * width * channels, Dtype(0), data_im);
        const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = height * width;

        for (int channel = channels; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                            data_col += output_w;
                        } else {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            for (int output_col = output_w; output_col; output_col--) {
                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    data_im[input_row * width + input_col] += *data_col;
                                }
                                data_col++;
                                input_col += stride_w;
                            }
                        }
                        input_row += stride_h;
                    }
                }
            }
        }
    }



} // namespace mkt
