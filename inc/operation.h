
namespace mkt {

    int gemm_nr(
        int trans_a, int trans_b,
        int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

    int bcnn_axpy(int n, float a, float *x, float *y);
}
