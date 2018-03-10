
#include "layer/softmax_layer.h"

namespace mkt {



    // Constructor with non parameter
    SoftmaxLayer::SoftmaxLayer(): pScale_{nullptr}, Layer(LayerType::Softmax) {};

    SoftmaxLayer::SoftmaxLayer(
        Layer* prevLayer,
        std::string id
    ): Layer(LayerType::Softmax)
    {
        id_ = id;
        batchSize_ = prevLayer->pDst_->NumOfData();
        pSrc_ = prevLayer->pDst_;

        int ih = prevLayer->pDst_->Height();
        int iw = prevLayer->pDst_->Width();
        int ic = prevLayer->pDst_->Channel();

        oh_ = ih;
        ow_ = iw;
        oc_ = ic;

        pDst_ = new Tensor{batchSize_, oh_, ow_, oc_};

        // pScale_ : for a plane of source data
        pScale_ = new Tensor{1, oh_, ow_, 1};
    };
    // Destructor
    SoftmaxLayer::~SoftmaxLayer(){
        delete pScale_;
    };

    /*
     * Initialize
     */
    void SoftmaxLayer::initialize() {
        initOutputTensor();

        // tensor for Scale data
        pScale_->allocate();
    };

    void SoftmaxLayer::Reshape(int num, int height, int width, int ch) {
        batchSize_ = num;
        oh_ = height;
        ow_ = width;
        oc_ = ch;

        if (pDst_)
        {
            // delete pDst_;
            pDst_->Reshape(num, height, width, ch);
        } else {
            pDst_ = new Tensor{batchSize_, oh_, ow_, oc_};
            pDst_->allocate();
        }

        if (pScale_)
        {
            // delete pScale_;
            pScale_->Reshape(num, height, width, ch);
        } else {
            pScale_ = new Tensor{1, oh_, ow_, 1};
            pScale_->allocate();
        }



    }

    /*
     * Computation Function
     */
    void SoftmaxLayer::Forward() {

        // For now we just use the channel as softmax axis

        float* pSrcData = pSrc_->cpu_data();
        int ic = pSrc_->Channel();
        int size2D = pSrc_->Size2D();
        int size3D = pSrc_->Size3D();

        float* pDstData = pDst_->cpu_data();
        fprintf(stderr, "pSrcData: %p\n", pSrcData);
        fprintf(stderr, "pDstData: %p\n", pDstData);
        mem_copy_cpu(pSrc_->WholeSize(), pSrcData, pDstData);
        // Scale data is used to
        // 1. store the maximum value of softmax axis
        // 2. store summation of exp(data)
        float* pScaleData = pScale_->cpu_data();

        // sum_multiplier is a Row vector (1 x ic) which has all elements are one
        Tensor sum_multiplier{1, 1, ic, 1}; // 1 X ic vector
        sum_multiplier.allocate();
        float* pSum_multiple_data = sum_multiplier.cpu_data();
        set_memory(sum_multiplier.WholeSize(), 1, pSum_multiple_data);

        for (int b = 0; b < batchSize_; ++b) {
            float* curPDstData = pDstData + b*size3D;

            // Copy one source data (previouse feature data)
            mem_copy_cpu(size2D, (pSrcData + b*size3D), pScaleData);

            // Get the maximum value of softmax axis
            for (int c = 0; c < ic; ++c)
            {
                for (int i = 0; i < size2D; ++i)
                {
                    pScaleData[i] = std::max(pScaleData[i], pSrcData[i + c*size2D + b*size3D]);
                }
            }

            // Subtraction
            // Before we take exponential, we have to substract the max to avoid
            // numerical issue
            // For example
            // M     = ic                     = 4
            // N     = Size2D = h X W = 5 * 4 = 20
            // K     = 1
            // Alpha = -1
            // Beta  = 1
            // A (ic     X 1) = pSum_multiple_data
            // B (Size2D X 1) = pScaleData
            //
            // AXB (ic X Size2D = Size3D)
            //
            // C = Alpha*(AXB) + Beta*C == C - (AXB)
            // Alpha *  (A  X             B          )
            //          |1|                              | scale_0, scale_1, ..., scale_19 |
            //  -1   *  |1| X [scale_0, ..., scale_19] = | scale_0, scale_1, ..., scale_19 | = M X N
            //          |1|              |               | scale_0, scale_1, ..., scale_19 |
            //          |1|              |               | scale_0, scale_1, ..., scale_19 |
            //           |               |
            //           |               |_________________
            //           |________________                 |
            //                            |                |
            // curPDstData = -1 * pSum_multiple_data X pScaleData + curPDstData
            //             = Dst data - max(Dst data)
            gemm_cpu(
                CblasTrans, CblasNoTrans,                   // trans_a, trans_b
                ic, size2D, 1,          // M, N, K
                -1,                     // Alpha
                pSum_multiple_data, ic, // A
                pScaleData, 1,          // B
                1,                      // Beta
                curPDstData, size2D     // C
            );

            // exp(curPDstData)
            for (int i = 0; i < size3D; ++i)
            {
                curPDstData[i] = std::exp(curPDstData[i]);
            }


            // sum exp(curPDstData)
            // gemv
            // y = Alpha * A X x + Beta * y
            //
            // Alpha = 1
            // Beta = 0
            // Y = A X x
            //            |a0_00, a0_01, a0_02, ..., a0_20|
            // A(4X20) =  |a1_00, a1_01, a1_02, ..., a1_20|
            //            |a2_00, a2_01, a2_02, ..., a2_20|
            //            |a3_00, a3_01, a3_02, ..., a3_20|
            //
            //            |1|
            // x(1xic) =  |1| (Row vector)
            //            |1|
            //            |1|
            //
            //                  |a0_00, a1_00, a2_00, a3_00|
            //                  |a0_01, a1_01, a2_01, a3_01|
            // Trans(A)(20x4) = |           .              |
            //                  |           .              |
            //                  |           .              |
            //                  |a0_20, a1_20, a2_20, a3_20|
            //
            //
            //                      |(a0_00 + a1_00 + a2_00 + a3_00)|
            //                |1|   |(a0_01 + a1_01 + a2_01 + a3_01)|
            // Y = Trans(A) X |1| = | .             .               |
            // |      |       |1|   | .             .               |
            // |      |       |1|   | .             .               |
            // |      |        |     |(a0_20 + a1_20 + a2_20 + a3_20)|
            // |      |        |
            // |      |        |_______________
            // |      |                        |
            // |      |__________              |
            // |____             |             |
            //      |            |             |
            // pScaleData = curPDstData X pSum_multiple_data + 0*pScaleData
            gemv_cpu(
                CBLAS_TRANSPOSE::CblasTrans, // trans_A
                ic,                 // M
                size2D,             // N
                1,                  // Alpha
                curPDstData,        // A
                pSum_multiple_data, // X
                0,                  // Beta
                pScaleData          // Y
            );

            // Division
            for (int c = 0; c < ic; ++c) {

                for (int sz = 0; sz < size2D; ++sz)
                {
                    curPDstData[sz + c*size2D] /= pScaleData[sz];
                }
            }
        }
    };

    void SoftmaxLayer::Backward() {

        // the derivative of softmax
        // dai/daj = aj(1-aj), if i == j ; i, j in N(number of softmax data)
        // dai/daj = ai*aj    , if i != j ; i, j in N(number of softmax data)


    };
}
