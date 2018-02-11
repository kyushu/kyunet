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
        id = id;

        int batchSize = prevLayer->pDst_->getNumOfData();
        int h = prevLayer->pDst_->getHeight();
        int w = prevLayer->pDst_->getWidth();
        int c = prevLayer->pDst_->getDepth();
        int size3D = prevLayer->pDst_->getSize3D();

        // pSrc_ point to pDst_ of previous layer
        pSrc_ = prevLayer->pDst_;

        pDst_ = new Tensor{batchSize, 1, unit, 1};
        pW_   = new Tensor{size3D, 1, 1, unit};
        pB_   = new Tensor{1, 1, unit, 1};

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
        int batchSize = prevLayer->pDst_->getNumOfData();
        int h = prevLayer->pDst_->getHeight();
        int w = prevLayer->pDst_->getWidth();
        int c = prevLayer->pDst_->getDepth();
        int size3D = prevLayer->pDst_->getSize3D();

        // pSrc_ point to pDst_ of previous layer
        pSrc_ = prevLayer->pDst_;

        pDst_ = new Tensor{batchSize, 1, unit, 1};
        pW_   = new Tensor{size3D, 1, 1, unit};
        pB_   = new Tensor{1, 1, unit, 1};

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

    void DenseLayer::forward() {
        fprintf(stderr, "##########################################\n");
        fprintf(stderr, "TODO: DenseLayer forward not yet finished\n");
        fprintf(stderr, "##########################################\n");

        float* pSrcData = pSrc_->getData();
        float* pDstData = pDst_->getData();
        float* pWData = pW_->getData();

        int batchSize = pSrc_->getNumOfData();
        int srcSize3D = pSrc_->getSize3D();
        int dstSize3D = pDst_->getSize3D();


        // 1. Rest data
        pDst_->cleanData();

        /****************************************************************
            2. Z =      X         x      Weight
                                          (N)
                                       (oh*ow*oc)
                       (K)          |w0 , ..., w15|
                    (ih*iw*ic)      |w16, ..., w23|
         (M)     | x0, ...,  x8|    |w24, ..., w31|   |z0 , ..., z15|
    (batch_size) | x9, ..., x16|  x |w32, ..., w39| = |z16, ..., z31|
                 |x17, ..., x24|    |w40, ..., w47|   |z32, ..., z47|
                                    |w48, ..., w55|
                                    |w56, ..., w63|
                                    |w64, ..., w71|
        ****************************************************************/
        gemm_cpu(0, 0,                                           /*trans_A, trans_B*/
            batchSize, dstSize3D, srcSize3D,  /*M, N, K*/
            1.0f, 1.0f,                                         /*ALPHA,   BETA*/
            pSrcData, srcSize3D,                      /*A,       lda(K)*/
            pWData,   dstSize3D,                      /*B,       ldb(N)*/
            pDstData, dstSize3D);                     /*C,       ldc(N)*/


        // 3. Z + bias
        addBias();

        // 4. A = next layer input = activation(Z)
        if (activationType_ != ActivationType::NONE)
        {
            pActivator_->forward(*pDst_, *pDst_);
        }

    }

    void DenseLayer::backward() {
        fprintf(stderr, "DenseLayer backward not yet finish\n");
    }
}
