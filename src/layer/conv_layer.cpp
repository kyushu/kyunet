#include "layer/conv_layer.h"

namespace mkt {

    // Constructor
    ConvLayer::ConvLayer(
        Layer* prevLayer,
        std::string id,
        int kernel_Height,
        int kernel_width,
        int kernel_channel,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        PaddingType paddingType,
        ActivationType actType,
        InitializerType weightInitType,
        InitializerType biasInitType
    ):
        // kernelSize_{kernelSize},
        fh_{kernel_Height},
        fw_{kernel_width},
        fc_{kernel_channel},
        stride_h_{stride_h},
        stride_w_{stride_w},
        pad_h_{pad_h},
        pad_w_{pad_w},
        paddingType_{paddingType},
        dilation_h_{1},
        dilation_w_{1},
        Layer(LayerType::Convolution, actType, weightInitType, biasInitType)
    {

        batchSize_ = prevLayer->pDst_->getNumOfData();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getDepth();
        // int size3D = prevLayer->pDst_->getSize3D();

        pSrc_ = prevLayer->pDst_;

        oc_ = fc_;
        // calculate pDst size: ow_ = (iw_ - fw_ +2padding_)/stride_ +1
        switch(paddingType_) {
            case PaddingType::valid:
                ow_ = static_cast<int>( static_cast<float>(iw - fw_ + 2*pad_w_) / stride_w_) + 1;
                oh_ = static_cast<int>( static_cast<float>(ih - fh_ + 2*pad_h_) / stride_h_) + 1;
                break;
            case PaddingType::same:

                pad_w_ = static_cast<int>( ((iw-1)*stride_w_ + fw_ - iw) * 0.5 );
                ow_    = static_cast<int>( static_cast<float>(iw - fw_ + (2*pad_w_)) / stride_w_ )  + 1;
                M_Assert(iw_ == ow_, "iw != ow");

                pad_h_ = static_cast<int>( ((ih-1)*stride_h_ + fh_ - ih) * 0.5 );
                oh_    = static_cast<int>( static_cast<float>(ih - fh_ + (2*pad_h_)) / stride_h_ ) + 1;
                M_Assert(ih_ == oh_, "ih != oh");

                break;
            default:
                ow_ = static_cast<int>( static_cast<float>(iw - fw_) / stride_w_)  + 1;
                oh_ = static_cast<int>( static_cast<float>(ih - fh_) / stride_h_)  + 1;
                break;
        }

        pDst_ = new Tensor{batchSize_, oh_, ow_, oc_};
        pW_   = new Tensor{ic, fh_, fw_, fc_};
        pB_   = new Tensor{1, 1, 1, oc_};

        pTmpCol_ = new Tensor{1, pW_->getSize2D()*ic, pDst_->getSize2D(), oc_};

        // Activator
        applyActivator();
    }

    // Destructor
    ConvLayer::~ConvLayer() {
        fprintf(stderr, "---------------------- ConvLayer Destructor\n");
        delete pTmpCol_;

    }


    // Initialization
    void ConvLayer::initialize() {

        initOutputTensor();
        initWeightTensor();
        initBiasTensor();

        // temporary memory for im2col and col2im
        pTmpCol_->allocate();

    }

    // Computation Function
    void ConvLayer::forward() {

        float* pSrcData = pSrc_->getData();
        float* pDstData = pDst_->getData();
        float* pWData = pW_->getData();
        float* pTmpColData = pTmpCol_->getData();

        int ic = pSrc_->getDepth();
        int iw = pSrc_->getWidth();
        int ih = pSrc_->getHeight();
        int src_size3D = pSrc_->getSize3D();
        int src_wholeSize = pSrc_->getWholeSize();

        int batchSize = pDst_->getNumOfData();
        int oh = pDst_->getHeight();
        int ow = pDst_->getWidth();
        int dst_wholeSize = pDst_->getWholeSize();

        int fh = pW_->getHeight();
        int fw = pW_->getWidth();
        int filter_wholeSize = pW_->getWholeSize();

        fprintf(stderr, "src_wholeSize: %d\n", src_wholeSize);
        fprintf(stderr, "dst_wholeSize: %d\n", dst_wholeSize);
        fprintf(stderr, "filter_wholeSize: %d\n", filter_wholeSize);

        // 1. Z = Conv(X)
        for (int i = 0; i < batchSize; ++i)
        {

            /*
                im2col
                For example:
                    src     =ih(3) X iw(3) X ic(3)
                    filter  = fh(2) X fw(2) X fc(2)
                    padding = 0
                    stride  = 1
                    dst     = oh(2) X ow(2)

                src =   | 0 1 2 | |  9 10 11 | | 18 19 20 |
                        | 3 4 5 | | 12 13 14 | | 21 22 23 |
                        | 6 7 8 | | 15 16 17 | | 24 25 26 |

                im2col(src) =   | s0   s1  s3  s4 |
                                | s1   s2  s4  s5 |  Channel 0
                                | s3   s4  s6  s7 |
                                | s4   s5  s7  s8 |____________
                                | s9  s10 s12 s13 |
                                | s10 s11 s13 s14 |  Channel 1
                                | s12 s13 s15 s16 |
                                | s13 s14 s16 s17 |____________
                                | s18 s19 s21 s22 |
                                | s19 s20 s22 s23 |  Channel 2
                                | s21 s22 s24 s25 |
                                | s22 s23 s25 s26 |____________
                                   |  |  |  |
                                   |  |  |  |_____________________
                                   |  |  |______________          |
                                   |  |________         |         |
                                   |           |        |         |
                            =   | patch_0 , patch_1, patch_2, patch_3 |

                im2col matrix = (fh*fw*ic) X (oh*ow)
            */
            mkt::im2col_cpu(pSrcData + i * src_size3D,
                ic, ih, iw,
                fh, fw,
                pad_h_, pad_w_,
                stride_h_, stride_w_,
                dilation_h_, dilation_w_,
                pTmpColData
            );



            /*
                GEMM: kernel X im2col
                For example: (same as above)
                    src     =ih(3) X iw(3) X ic(3)
                    filter  = fh(2) X fw(2) X fc(2)
                    padding = 0
                    stride  = 1
                    dst     = oh(2) X ow(2)

                Step 1. treat filter as [w0, w1, w2, w3] (fhxfw)
                        so filter matrix(W) is
                    | w0_0, ..., w0_3, w0_4, ..., w0_7, w0_8, ..., w0_11 | = filter 0 (F0)
                    | w1_0, ..., w1_3, w1_4, ..., w1_7, w1_8, ..., w0_11 | = filter 1 (F1)

                    W matrix = oc X (fh*fw*ic)


                Step 2. Dst matrix = W X im2col(src) =

                                | s0   s1  s3  s4 |
                                | s1   s2  s4  s5 |
                                | s3   s4  s6  s7 |
                                | s4   s5  s7  s8 |
                                | s9  s10 s12 s13 |
                    | F0 |      | s10 s11 s13 s14 |
                    | F1 |  X   | s12 s13 s15 s16 |
                                | s13 s14 s16 s17 |
                                | s18 s19 s21 s22 |
                                | s19 s20 s22 s23 |
                                | s21 s22 s24 s25 |
                                | s22 s23 s25 s26 |
     |
                    Dst matrix = oc X (oh*ow)

            */

            mkt::gemm_cpu(
                0, 0,                                                           /*trans_A, trans_B*/
                pW_->getDepth(), pDst_->getSize2D(), pW_->getSize2D()*ic,   /*M,       N, K*/
                1.0f, 1.0f,                                                     /* ALPHA,   BETA */
                pWData, pW_->getSize2D()*ic,                               /*A,       lda(K)*/
                pTmpColData,   oh*ow,                                      /*B,       ldb(N)*/
                pDstData, oh*ow                                            /*C,       ldc(N)*/
            );
        }


        // 2. Z + bias
        addBias();

        // 3. A = next layer input = activation(Z)
        if (activationType_ != ActivationType::NONE)
        {
            pActivator_->forward(*pDst_, *pDst_);
        }

    }
    void ConvLayer::backward() {

    }


    // Getter
    int ConvLayer::getFilterHeight() {
        return fh_;
    }
    int ConvLayer::getFilterWidth() {
        return fw_;
    }
    int ConvLayer::getFilterChannel() {
        return fc_;
    }
}
