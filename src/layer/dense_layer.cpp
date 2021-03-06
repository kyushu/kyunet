#include "layer/dense_layer.h"
#include "test_utils.hpp"

namespace mkt {

    // Constructor with ID
    DenseLayer::DenseLayer(
        Layer* prevLayer,
        std::string id,
        int unit,
        ActivationType actType,
        InitializerType weightInitType,
        InitializerType biasInitType
    ): unit_{unit}, Layer(LayerType::DENSE, actType, weightInitType, biasInitType)
    {
        id_ = id;

        batchSize_ = prevLayer->pDst_->getNumOfData();
        int h = prevLayer->pDst_->getHeight();
        int w = prevLayer->pDst_->getWidth();
        int c = prevLayer->pDst_->getChannel();
        int input_size3D = prevLayer->pDst_->getSize3D();

        pPrevLayer_ = prevLayer;

        pDst_ = new Tensor{batchSize_, 1, 1, unit};
        pgDst_ = new Tensor{batchSize_, 1, 1, unit};

        pW_   = new Tensor{1, unit, input_size3D, 1};
        pgW_  = new Tensor{1, unit, input_size3D, 1};

        pB_   = new Tensor{1, 1, 1, unit};
        pgB_  = new Tensor{1, 1, 1, unit};

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
    ): unit_{unit}, Layer(LayerType::DENSE, actType, weightInitType, biasInitType)
    {
        batchSize_ = prevLayer->pDst_->getNumOfData();
        int h = prevLayer->pDst_->getHeight();
        int w = prevLayer->pDst_->getWidth();
        int c = prevLayer->pDst_->getChannel();
        int input_size3D = prevLayer->pDst_->getSize3D();

        pPrevLayer_ = prevLayer;

        pDst_  = new Tensor{batchSize_, 1, 1, unit_};
        pgDst_ = new Tensor{batchSize_, 1, 1, unit_};

        pW_   = new Tensor{1, unit_, input_size3D, 1};
        pgW_  = new Tensor{1, unit_, input_size3D, 1};

        pB_   = new Tensor{1, 1, 1, unit_};
        pgB_  = new Tensor{1, 1, 1, unit_};

        // Activator
        applyActivator();
    }

    DenseLayer::DenseLayer(Layer* prevLayer, std::string id, LayerParams params):Layer(LayerType::DENSE) {

        id_ = id;

        batchSize_ = prevLayer->pDst_->getNumOfData();
        int h = prevLayer->pDst_->getHeight();
        int w = prevLayer->pDst_->getWidth();
        int c = prevLayer->pDst_->getChannel();
        int input_size3D = prevLayer->pDst_->getSize3D();

        pPrevLayer_ = prevLayer;

        // Parameter setting
        activationType_ = params.actType;
        weightInitType_ = params.weight_init_type;
        biasInitType_   = params.bias_init_type;

        unit_ = params.fc;

        pDst_ = new Tensor{batchSize_, 1, 1, unit_};
        pgDst_ = new Tensor{batchSize_, 1, 1, unit_};

        pW_   = new Tensor{1, unit_, input_size3D, 1};
        pgW_  = new Tensor{1, unit_, input_size3D, 1};

        pB_   = new Tensor{1, 1, 1, unit_};
        pgB_  = new Tensor{1, 1, 1, unit_};

        // Activator
        applyActivator();
    }

    /* Destructor */
    DenseLayer::~DenseLayer(){
        fprintf(stderr, "--------------------- denseLayer Destructor\n");
    }

    void DenseLayer::initialize(NetMode mode) {

        MKT_Assert(pDst_ != nullptr, "pDst_ is null");
        MKT_Assert(pgDst_ != nullptr, "pgDst_ is null");
        MKT_Assert(pW_ != nullptr, "pW_ is null");
        MKT_Assert(pgW_ != nullptr, "pgW_ is null");
        MKT_Assert(pB_ != nullptr, "pB_ is null");
        MKT_Assert(pgB_ != nullptr, "pgB_ is null");

        MKT_Assert(pActivator_ != nullptr, "pActivator_ is null");

        initOutputTensor();
        initWeightTensor();
        initBiasTensor();

        initGradTensor();
        initGradWeightTensor();
        initGradBiasTensor();
    }

    void DenseLayer::Forward() {

        // 1. Rest data
        pDst_->resetData();
        pgDst_->resetData();
        pgW_->resetData();
        pgB_->resetData();

        Tensor* pSrc = pPrevLayer_->pDst_;

        // int batchSize = pSrc->getNumOfData();

        float* pSrcData = pSrc->getCPUData();
        int srcSize3D = pSrc->getSize3D();

        float* pDstData = pDst_->getCPUData();
        int dstSize3D = pDst_->getSize3D();

        float* pWData = pW_->getCPUData();


        gemm_cpu(CblasNoTrans, CblasTrans,      /* trans_A, trans_B  */
            batchSize_, dstSize3D, srcSize3D,   /* M, N, K           */
            1.0f,                               /* ALPHA             */
            pSrcData, srcSize3D,                /* A,       lda(K)   */
            pWData,   srcSize3D,                /* B,       ldb(K)   */
            1.0f,                               /* BETA             */
            pDstData, dstSize3D);               /* C,       ldc(N)   */

        // 3. Z + bias
        // addBias();

        // 4. A = next layer input = activation(Z)
        int fc_dst_c = pDst_->getChannel();
        int fc_dst_h = pDst_->getHeight();
        int fc_dst_w = pDst_->getWidth();

        if (activationType_ != ActivationType::NONE)
        {
            pActivator_->Forward(*pDst_, *pDst_);
        }
    }

    void DenseLayer::Backward() {

        float* pWData = pW_->getCPUData();
        int dstSize3D = pDst_->getSize3D();

        float* pgDstData = pgDst_->getCPUData();
        int gDst_size3D = pgDst_->getSize3D();

        // 1. Back from Activator first
        if (activationType_ != ActivationType::NONE)
        {
            pActivator_->Backward(*pDst_, *pgDst_, *pgDst_);
        }

        // 2. [Update gradient with respect to Bias]
        // dL/db = d^(l+1)
        float* pgBData = pgB_->getCPUData();
        for (int i = 0; i < batchSize_; ++i)
        {
            axpy(gDst_size3D, 1.0f, pgDstData + i * gDst_size3D, pgBData);
        }

        // 3. [Update gradient with respect to Weight]
        /*************************************************************
         * dL/dw = d^(l+1) * src_data
         * For Dense layer, Size2D = Size3D (Channel = 1)
         *
         * A = pgDstData (M x N) = (Batch_size x Dst_Size3D)
         * B = pSrcData  (M x K) = (Batch_size x Src_Size3D)
         * Tranpose(A) = (N x M) = (Dst_Size3D x Batch_size)
         * C = pgWData   (N x K) = (Dst_Size3D x Src_Size3D)
         * C = A*B+C = (N xM) * (M x K) + (N x K) = (N x K) + (N x K)
         *************************************************************/
        float* pgWData = pgW_->getCPUData();

        float* pSrcData = pPrevLayer_->pDst_->getCPUData();
        int src_size3D = pPrevLayer_->pDst_->getSize3D();

        int dst_size3D = pDst_->getSize3D();
        // pgWData will be reset after update weight
        gemm_cpu(
            CblasTrans, CblasNoTrans,
            dst_size3D, src_size3D, batchSize_,
            1.0f,
            pgDstData, dst_size3D,
            pSrcData, src_size3D,
            1.0f,
            pgWData, src_size3D
        );

        // 4. [Update gradient with respect to data]
        /*********************************************************************************
         *
         * For example,
         * M = Batch Size = 3
         * N = Dst_Size2D = 9  (previous layer output size)
         * K = Src_Size2D = 16 (output size of this layer)
         *                                    (N=9)
         *                       |w_00_0, w_00_01, w_00_02, ...,  w_00_08|
         *                       |w_01_0, w_01_01, w_01_03, ...,  w_01_08|
         *         (K=16)        |w_02_0, w_02_01, w_02_03, ...,  w_02_08|        (N=9)
         *   |gz00, ..., gz15|   |w_03_0, w_03_01, w_03_03, ...,  w_03_08|   |gx00, ..., gx08|
         *   |gz16, ..., gz31| = |w_04_0, w_04_01, w_04_03, ...,  w_04_08| = |gx09, ..., gx16|
         *   |gz32, ..., gz47|   |                   .                   |   |gx17, ..., gx24|
         *                       |                   .                   |
         *                       |                   .                   |
         *                       |w_15_00, w_15_01, w_15_03, ..., w_15_08|
         *
         * A = pgDstData (M x N)
         * B = pWdata    (N x K)
         * C = pSrc_gData = A * B = (M x N) (N x K) = (M x K)
         **********************************************************************************/
        if (pPrevLayer_->pgDst_)
        {
            float* pSrc_gData = pPrevLayer_->pgDst_->getCPUData();
            int srcSize3D = pPrevLayer_->pgDst_->getSize3D();

            // pSrc_dif = pgDstData * pWdata + pSrc_dif
            gemm_cpu(
                CblasNoTrans, CblasNoTrans,
                batchSize_, srcSize3D, dstSize3D,
                1.0f,
                pgDstData, gDst_size3D,
                pWData, src_size3D,
                0,
                pSrc_gData, src_size3D
            );
        }
    }
}


/**
 * #### [ Forward  Pass] ####
 * For example: M=3, N=16, K=9
 *
 * For Dense layer, Size2D = Size3D (Channel = 1)
 *
 * 2. Z =             A(x)      x                B(Weight)
 *                                                (N=16)
 *                                             (oh*ow*oc = 2*2*4 = 16)
 *                 (K = 9)
 *             (ih*iw*ic = 3*3*3 = 9)
 *                                           |w_00_00, ...,  w_0_08|
 *                                           |w_01_00, ...,  w_1_08|
 *      (M)     | x0, ...,  x8|              |w_02_00, ...,  w_2_08|     |z0 , ..., z15|
 * (batch_size) | x9, ..., x16| x trapnspos( |w_03_00, ...,  w_3_08| ) = |z16, ..., z31|
 *              |x17, ..., x24|              |w_04_00, ...,  w_4_08|     |z32, ..., z47|
 *                                           |                     |
 *                                           |                     |
 *                                           |                     |
 *                                           |w_15_00, ..., w_15_08|
 *                                                    ||
 *                                                   (16)
 *                                       |w_00_00, w_01_00, ..., w_15_00|
 *                                       |w_00_01, w_01_01, ..., w_15_01|
 *                                       |w_00_02, w_01_02, ..., w_15_02|
 *                                       |w_00-03, w_01_03, ..., w_15_03|
 *                                       |w_00_04, w_01_04, ..., w_15_04| (9)
 *                                       |w_00_05, w_01_05, ..., w_15_05|
 *                                       |w_00_06, w_01_06, ..., w_15_06|
 *                                       |w_00_07, w_01_07, ..., w_15_07|
 *                                       |w_00_08, w_01_08, ..., w_15_08|
 * w_n1_n2
 * n1: index of destination
 * n2: index of source
 *
 *
 *
 * #### [ Backpropagation ] ####
 * 1. Back from Activator first
 * 2. [Update gradient with respect to Bias]
 *  dL/db = d^(l+1)
 *
 * 3. [Update gradient with respect to Weight]
 *  dL/dw = d^(l+1) * src_data
 *  For Dense layer, Size2D = Size3D (Channel = 1)
 *
 *  A = pgDstData (M x N) = (Batch_size x Dst_Size3D)
 *  B = pSrcData  (M x K) = (Batch_size x Src_Size3D)
 *  Tranpose(A) = (N x M) = (Dst_Size3D x Batch_size)
 *  C = pgWData   (N x K) = (Dst_Size3D x Src_Size3D)
 *  C = A*B+C = (N xM) * (M x K) + (N x K) = (N x K) + (N x K)
 *
 * 4. [Update gradient with respect to data]
 *  For example,
 *  M = Batch Size = 3
 *  N = Dst_Size2D = 9  (previous layer output size)
 *  K = Src_Size2D = 16 (output size of this layer)
 *                                        (N=9)
 *                        |w_00_0, w_00_01, w_00_02, ...,  w_00_08|
 *                        |w_01_0, w_01_01, w_01_03, ...,  w_01_08|
 *          (K=16)        |w_02_0, w_02_01, w_02_03, ...,  w_02_08|        (N=9)
 *    |gz00, ..., gz15|   |w_03_0, w_03_01, w_03_03, ...,  w_03_08|   |gx00, ..., gx08|
 *    |gz16, ..., gz31| = |w_04_0, w_04_01, w_04_03, ...,  w_04_08| = |gx09, ..., gx16|
 *    |gz32, ..., gz47|   |                   .                   |   |gx17, ..., gx24|
 *                        |                   .                   |
 *                        |                   .                   |
 *                        |w_15_00, w_15_01, w_15_03, ..., w_15_08|
 *
 *  A = pgDstData (M x N)
 *  B = pWdata    (N x K)
 *  C = pSrc_gData = A * B = (M x N) (N x K) = (M x K)
 */
