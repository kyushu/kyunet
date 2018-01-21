#include "denseLayer.h"

namespace mkt {

    void DenseLayer::initialize() {
        initOutputTensor();
        initWeightTensor(initType);
        initBiasTensor(initType);

    }

    void DenseLayer::forward() {
        fprintf(stderr, "##########################################\n");
        fprintf(stderr, "TODO: DenseLayer forward not yet finished\n");
        fprintf(stderr, "##########################################\n");

        // Rest data
        pDst->cleanData();

        // Z = X * Weight
        gemm_nr(0, 0,
            pDst->batchSize, pDst->size3D, pSrc->size3D,    /*M,N,K*/
            1.0f, 1.0f,                                     /*ALPHA, BETA*/
            pSrc->pData, pSrc->size3D,                      /*A, lda(K)*/
            pW->pData, pDst->size3D,                        /*B, ldb(N)*/
            pDst->pData, pDst->size3D);                     /*C, ldc(N)*/


        // Z + bias
        addBias();

        // A = activation(Z)

    }

    void DenseLayer::backward() {
        fprintf(stderr, "DenseLayer backward not yet finish\n");
    }
}
