#include "denseLayer.h"

namespace mkt {

    void DenseLayer::initialize() {
        initOutputTensor();
        initWeightTensor(weightInitType);
        initBiasTensor(biasInitType);

    }

    void DenseLayer::forward() {
        fprintf(stderr, "##########################################\n");
        fprintf(stderr, "TODO: DenseLayer forward not yet finished\n");
        fprintf(stderr, "##########################################\n");

        // 1. Rest data
        pDst->cleanData();

        // 2. Z = X * Weight
        gemm_nr(0, 0,                                       /*trans_A, trans_B*/
            pDst->batchSize, pDst->size3D, pSrc->size3D,    /*M,       N,K*/
            1.0f, 1.0f,                                     /*ALPHA,   BETA*/
            pSrc->pData, pSrc->size3D,                      /*A,       lda(K)*/
            pW->pData,   pDst->size3D,                      /*B,       ldb(N)*/
            pDst->pData, pDst->size3D);                     /*C,       ldc(N)*/


        // 3. Z + bias
        addBias();

        // 4. A = next layer input = activation(Z)
        applyActivation(); // decided by activationType

    }

    void DenseLayer::backward() {
        fprintf(stderr, "DenseLayer backward not yet finish\n");
    }
}
