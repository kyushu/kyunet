
namespace mkt {

    int gemm_nr(
        int trans_a, int trans_b,
        int M, int N, int K,
        float ALPHA, float BETA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc);

    //
    int axpy(int n, float a, float *x, float *y);
}
