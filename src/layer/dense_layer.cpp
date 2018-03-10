#include "layer/dense_layer.h"

namespace mkt {

    // Constructor with ID
    DenseLayer::DenseLayer(
        Layer* prevLayer,
        std::string id,
        int unit,
        ActivationType actType,
        InitializerType weightInitType,
        InitializerType biasInitType
    ): unit_{unit}, Layer(LayerType::FullConnected, actType, weightInitType, biasInitType)
    {
        id_ = id;

        int batchSize = prevLayer->pDst_->NumOfData();
        int h = prevLayer->pDst_->Height();
        int w = prevLayer->pDst_->Width();
        int c = prevLayer->pDst_->Channel();
        int size3D = prevLayer->pDst_->Size3D();

        // pSrc_ point to pDst_ of previous layer
        pSrc_ = prevLayer->pDst_;

        pDst_ = new Tensor{batchSize, 1, 1, unit};
        pW_   = new Tensor{1, size3D, unit, 1};
        pB_   = new Tensor{1, 1, 1, unit};

        // Activator
        applyActivator();
    }

    // Constructor without ID
    DenseLayer::DenseLayer(
        Layer* prevLayer,
        int unit,
        ActivationType actType,
        InitializerType weightInitType,
        InitializerType biasInitType
    ): unit_{unit}, Layer(LayerType::FullConnected, actType, weightInitType, biasInitType)
    {
        int batchSize = prevLayer->pDst_->NumOfData();
        int h = prevLayer->pDst_->Height();
        int w = prevLayer->pDst_->Width();
        int c = prevLayer->pDst_->Channel();
        int size3D = prevLayer->pDst_->Size3D();

        // pSrc_ point to pDst_ of previous layer
        pSrc_ = prevLayer->pDst_;

        pDst_ = new Tensor{batchSize, 1, 1, unit};
        pW_   = new Tensor{size3D, 1, 1, unit};
        pB_   = new Tensor{1, 1, 1, unit};

        // Activator
        applyActivator();
    }

    /* Destructor */
    DenseLayer::~DenseLayer(){
        fprintf(stderr, "--------------------- denseLayer Destructor\n");
    }

    void DenseLayer::initialize() {
        initOutputTensor();
        initWeightTensor();
        initBiasTensor();

    }

    void DenseLayer::Forward() {

        float* pSrcData = pSrc_->cpu_data();
        float* pDstData = pDst_->cpu_data();
        float* pWData = pW_->cpu_data();

        int batchSize = pSrc_->NumOfData();
        int srcSize3D = pSrc_->Size3D();
        int dstSize3D = pDst_->Size3D();


        // 1. Rest data
        pDst_->cleanData();

        /****************************************************************
            2. Z =      X         x      Weight
                                          (N)
                                       (oh*ow*oc)
                       (K)
                    (ih*iw*ic)      |w0 , ..., w15|
                                    |w16, ..., w23|
         (M)     | x0, ...,  x8|    |w24, ..., w31|   |z0 , ..., z15|
    (batch_size) | x9, ..., x16|  x |w32, ..., w39| = |z16, ..., z31|
                 |x17, ..., x24|    |w40, ..., w47|   |z32, ..., z47|
                                    |w48, ..., w55|
                                    |w56, ..., w63|
                                    |w64, ..., w71|
        ****************************************************************/
        gemm_cpu(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,                              /*trans_A, trans_B*/
            batchSize, dstSize3D, srcSize3D,        /*M, N, K*/
            1.0f,                              /*ALPHA*/
            pSrcData, srcSize3D,                    /*A,       lda(K)*/
            pWData,   dstSize3D,                    /*B,       ldb(N)*/
            1.0f,                                   /* BETA*/
            pDstData, dstSize3D);                   /*C,       ldc(N)*/


        // 3. Z + bias
        addBias();

        // 4. A = next layer input = activation(Z)
        if (activationType_ != ActivationType::NONE)
        {
            pActivator_->forward(*pDst_, *pDst_);
        }

    }

    void DenseLayer::Backward() {
        fprintf(stderr, "DenseLayer backward not yet finish\n");
    }
}
