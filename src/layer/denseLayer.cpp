#include "denseLayer.h"

namespace mkt {

    void DenseLayer::initialize() {
        initOutputTensor();
        initWeightTensor(weightInitType_);
        initBiasTensor(biasInitType_);

    }

    void DenseLayer::forward() {
        fprintf(stderr, "##########################################\n");
        fprintf(stderr, "TODO: DenseLayer forward not yet finished\n");
        fprintf(stderr, "##########################################\n");

        // 1. Rest data
        pDst_->cleanData();

        // 2. Z = X * Weight
        gemm_cpu(0, 0,                                           /*trans_A, trans_B*/
            pDst_->num_, pDst_->size3D_, pSrc_->size3D_,  /*M,       N,K*/
            1.0f, 1.0f,                                         /*ALPHA,   BETA*/
            pSrc_->pData_, pSrc_->size3D_,                      /*A,       lda(K)*/
            pW_->pData_,   pDst_->size3D_,                      /*B,       ldb(N)*/
            pDst_->pData_, pDst_->size3D_);                     /*C,       ldc(N)*/


        // 3. Z + bias
        addBias();

        // 4. A = next layer input = activation(Z)
        applyActivation(); // decided by activationType

    }

    void DenseLayer::backward() {
        fprintf(stderr, "DenseLayer backward not yet finish\n");
    }
}
