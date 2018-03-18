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

        batchSize_ = prevLayer->pDst_->NumOfData();
        int h = prevLayer->pDst_->Height();
        int w = prevLayer->pDst_->Width();
        int c = prevLayer->pDst_->Channel();
        int input_size3D = prevLayer->pDst_->Size3D();

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
    ): unit_{unit}, Layer(LayerType::FullConnected, actType, weightInitType, biasInitType)
    {
        batchSize_ = prevLayer->pDst_->NumOfData();
        int h = prevLayer->pDst_->Height();
        int w = prevLayer->pDst_->Width();
        int c = prevLayer->pDst_->Channel();
        int input_size3D = prevLayer->pDst_->Size3D();

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

    DenseLayer::DenseLayer(Layer* prevLayer, std::string id, LayerParams params):Layer(LayerType::FullConnected) {

        id_ = id;

        batchSize_ = prevLayer->pDst_->NumOfData();
        int h = prevLayer->pDst_->Height();
        int w = prevLayer->pDst_->Width();
        int c = prevLayer->pDst_->Channel();
        int input_size3D = prevLayer->pDst_->Size3D();

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
    }

    /* Destructor */
    DenseLayer::~DenseLayer(){
        fprintf(stderr, "--------------------- denseLayer Destructor\n");
    }

    void DenseLayer::initialize() {
        initOutputTensor();
        initWeightTensor();
        initBiasTensor();

        initGradOutputTensor();
        initGradWeightTensor();
        initGradBiasTensor();

    }

    void DenseLayer::Forward() {

        Tensor* pSrc = pPrevLayer_->pDst_;

        // int batchSize = pSrc->NumOfData();

        float* pSrcData = pSrc->cpu_data();
        int srcSize3D = pSrc->Size3D();

        float* pDstData = pDst_->cpu_data();
        int dstSize3D = pDst_->Size3D();

        float* pWData = pW_->cpu_data();

        // 1. Rest data
        pDst_->cleanData();

        /********************************************************************
         * For example: M=3, N=16, K=9
         * 2. Z =             A(x)      x                B(Weight)
         *                                                (N=16)
         *                                             (oh*ow*oc = 2*2*4 = 16)
         *                 (K = 9)
         *             (ih*iw*ic = 3*3*3 = 9)
         *                                           |w00_0, ...,  w0_8|
         *                                           |w01_0, ...,  w1_8|
         *      (M)     | x0, ...,  x8|              |w02_0, ...,  w2_8|     |z0 , ..., z15|
         * (batch_size) | x9, ..., x16| x trapnspos( |w03_0, ...,  w3_8| ) = |z16, ..., z31|
         *              |x17, ..., x24|              |w04_0, ...,  w4_8|     |z32, ..., z47|
         *                                           |                 |
         *                                           |                 |
         *                                           |                 |
         *                                           |w15_0, ..., w15_8|
         *                                                     ||
         *                                                    (16)
         *                                              |w00, ...,  w15|
         *                                              |w16, ...,  w31|
         *                                              |w32, ...,  w47|
         *                                              |w48, ...,  w63| (9)
         *                                              |w64, ...,  w79|
         *                                              |w80, ...,  w95|
         *                                              |w96, ..., w111|
         *                                              |w112,..., w127|
         *
         ********************************************************************/
        gemm_cpu(CblasNoTrans, CblasTrans,      /*trans_A, trans_B*/
            batchSize_, dstSize3D, srcSize3D,   /*M, N, K*/
            1.0f,                               /*ALPHA*/
            pSrcData, srcSize3D,                /*A,       lda(K)*/
            pWData,   srcSize3D,                /*B,       ldb(K)*/
            1.0f,                               /* BETA*/
            pDstData, dstSize3D);               /*C,       ldc(N)*/

        // 3. Z + bias
        // addBias();

        // 4. A = next layer input = activation(Z)
        if (activationType_ != ActivationType::NONE)
        {
            pActivator_->Forward(*pDst_, *pDst_);
        }

    }

    void DenseLayer::Backward() {
        fprintf(stderr, "DenseLayer backward not yet finish\n");

        float* pWData = pW_->cpu_data();
        int dstSize3D = pDst_->Size3D();

        float* pgDstData = pgDst_->cpu_data();
        int gDst_size3D = pgDst_->Size3D();

        // 1. Back from Activator first
        if (activationType_ != ActivationType::NONE)
        {
            pActivator_->Backward(*pDst_, *pgDst_, *pgDst_);
        }


        // 2. [Update gradient with respect to Weight]
        /*************************************************************
         *
         * A = pgDstData (M x N) = (Batch_size x Dst_Size3D)
         * B = pSrcData  (M x K) = (Batch_size x Src_Size3D)
         * T(A) =        (N x M) = (Dst_Size3D x Batch_size)
         * C = pgWData    (N x K) = (Dst_Size3D x Src_Size3D)
         * C = A*B+C = (N xM) * (M x K) + (N x K) = (N x K) + (N x K)
         *************************************************************/
        float* pgWData = pgW_->cpu_data();

        float* pSrcData = pPrevLayer_->pDst_->cpu_data();
        int src_size3D = pPrevLayer_->pDst_->Size3D();

        int dst_size3D = pDst_->Size3D();
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


        // 3. [Update gradient with respect to Bias]
        float* pgBData = pgB_->cpu_data();
        for (int i = 0; i < batchSize_; ++i)
        {
            axpy(dst_size3D, 1.0f, pgDstData + i * dst_size3D, pgBData);
        }


        // 4. [Update gradient with respect to data]
        /*********************************************************************************
         *
         * For example,
         * M = Batch Size = 3
         * N = Dst_Size3D = 9
         * K = Src_Size3D = 16
         *                                    (N=9)
         *                       |w00_0, w00_1, w00_2, ...,  w0_8|
         *                       |w01_0, w01_1, w01_3, ...,  w1_8|
         *         (K=16)        |w02_0, w02_1, w02_3, ...,  w2_8|        (N=9)
         *   |gz00, ..., gz15|   |w03_0, w03_1, w03_3, ...,  w3_8|   |gx00, ..., gx08|
         *   |gz16, ..., gz31| = |w04_0, w04_1, w04_3, ...,  w4_8| = |gx09, ..., gx16|
         *   |gz32, ..., gz47|   |              .                |   |gx17, ..., gx24|
         *                       |              .                |
         *                       |              .                |
         *                       |w15_0, w15_1, w15_3, ..., w15_8|
         *
         * A = pgDstData (M x N)
         * B = pWdata    (N x K)
         * C = pSrc_gData = A * B = (M x N) (N x K) = (M x K)
         **********************************************************************************/
        if (pPrevLayer_->pgDst_)
        {
            float* pSrc_gData = pPrevLayer_->pgDst_->cpu_data();
            int srcSize3D = pPrevLayer_->pgDst_->Size3D();

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
