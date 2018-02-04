#include "denseLayer.h"

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
        pW_   = new Tensor{size3D, unit, 1, 1};
        pB_   = new Tensor{1, 1, unit, 1};

        // TODO: Activation setting
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
        pW_   = new Tensor{1, size3D, unit, 1};
        pB_   = new Tensor{1, 1, unit, 1};

        // TODO: Activation
    }

    // Destructor
    DenseLayer::~DenseLayer(){
        fprintf(stderr, "--------------------- denseLayer Destructor\n");
    }

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
