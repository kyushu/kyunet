
#include "layer/softmax_layer.h"

namespace mkt {

    // Constructor with non parameter
    template<typename T>
    SoftmaxLayer<T>::SoftmaxLayer(): pScale_{nullptr}, Layer<T>(LayerType::SOFTMAX) {};

    template<typename T>
    SoftmaxLayer<T>::SoftmaxLayer(
        Layer<T>* prevLayer,
        std::string id
    ): Layer<T>(LayerType::SOFTMAX)
    {
        this->id_ = id;
        this->batchSize_ = prevLayer->pDst_->getNumOfData();

        this->pPrevLayer_ = prevLayer;

        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

        this->oh_ = ih;
        this->ow_ = iw;
        this->oc_ = ic;

        this->pDst_  = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };
        this->pgDst_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };

        // pScale_ : for a plane of source data
        pScale_ = new Tensor<T>{ 1, this->oh_, this->ow_, 1 };
    };

    // Destructor
    template<typename T>
    SoftmaxLayer<T>::~SoftmaxLayer(){
        delete pScale_;
    };

    /*
     * Initialize
     */
    template<typename T>
    void SoftmaxLayer<T>::initialize(NetMode mode) {

        MKT_Assert(this->pDst_ != nullptr, "pDst_ is null");
        MKT_Assert(this->pgDst_ != nullptr, "pgDst_ is null");
        MKT_Assert(pScale_ != nullptr, "pScale_ is null");

        this->initOutputTensor();
        this->initGradTensor();

        // tensor for Scale data
        pScale_->allocate();
    };

    template<typename T>
    void SoftmaxLayer<T>::Reshape(int num, int height, int width, int ch) {
        this->batchSize_ = num;
        this->oh_ = height;
        this->ow_ = width;
        this->oc_ = ch;

        if (this->pDst_)
        {
            this->pDst_->Reshape(num, height, width, ch);
        } else {
            this->pDst_ = new Tensor<T>{ this->batchSize_, this->oh_, this->ow_, this->oc_ };
            this->pDst_->allocate();
        }

        if (pScale_)
        {
            // delete pScale_;
            pScale_->Reshape(num, height, width, ch);
        } else {
            pScale_ = new Tensor<T>{ 1, this->oh_, this->ow_, 1 };
            pScale_->allocate();
        }

    }

    /*
     * Computation Function
     */
    template<typename T>
    void SoftmaxLayer<T>::Forward() {

        // For now we just use the channel as softmax axis
        Tensor<T>* pSrc = this->pPrevLayer_->pDst_;

        T* pSrcData = pSrc->getCPUData();
        int ic = pSrc->getChannel();
        int size2D = pSrc->getSize2D();
        int size3D = pSrc->getSize3D();

        T* pDstData = this->pDst_->getCPUData();
        mem_copy_cpu(pSrc->getWholeSize(), pSrcData, pDstData);
        // Scale data is used to
        // 1. store the maximum value of softmax axis
        // 2. store summation of exp(data)
        T* pScaleData = pScale_->getCPUData();

        // sum_multiplier is a Row vector (1 x ic) which has all elements are one
        Tensor<T> sum_multiplier{ 1, 1, ic, 1 }; // 1 X ic vector
        sum_multiplier.allocate();
        T* pSum_multiple_data = sum_multiplier.getCPUData();
        set_memory(sum_multiplier.getWholeSize(), 1, pSum_multiple_data);

        for (int b = 0; b < this->batchSize_; ++b) {
            T* curPDstData = pDstData + b*size3D;

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

    template<typename T>
    void SoftmaxLayer<T>::Backward() {

        // the derivative of softmax
        // dai/daj = aj(1-aj), if i == j ; i, j in N(number of softmax data)
        // dai/daj = ai*aj    , if i != j ; i, j in N(number of softmax data)
        fprintf(stderr, "%s, %s, %d not yet implemented\n", __FILE__, __func__, __LINE__);
    };

    // Explicitly instantiate the template, and its member definitions
    template class SoftmaxLayer<float>;

} // namespace mkt
