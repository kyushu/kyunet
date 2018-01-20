#include "denseLayer.h"

namespace mkt {

    void DenseLayer::initialize() {
        initOutputTensor();
        initWeightTensor(Initializer_Type::NONE);
        initBiasTensor(Initializer_Type::NONE);

    }

    void DenseLayer::forward() {
        fprintf(stderr, "DenseLayer forward not yet finished\n");

        // Rest data
        pDst->cleanData();

        // Z = X * Weight
        gemm_nr(0, 0,
            pDst->batchSize, unit, pSrc->size3D, 1.0f,
            pSrc->pData, pSrc->size3D,
            pW->pData, unit,
            1.0f,
            pDst->pData, pDst->size3D);


        // Z + bias
        axpy(pDst->size3D, 1, pB->pData, pDst->pData);

    }

    void DenseLayer::backward() {
        fprintf(stderr, "DenseLayer backward not yet finish\n");
    }
}
