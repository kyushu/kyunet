#include "layer/conv_layer.h"

namespace mkt {

    // Constructor
    ConvLayer::ConvLayer(
            Layer* prevLayer,
            std::string id,
            int nfilter, int kernelSize, int stride,
            int padding, PaddingType paddingType,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
    ):
        nfilter_{nfilter},
        kernelSize_{kernelSize},
        stride_{stride},
        padding_{padding},
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

        oc_ = nfilter;
        // calculate pDst size: W_o = (W-f +2p)/s +1
        switch(paddingType_) {
            case PaddingType::valid:
                ow_ = floor(float(iw - kernelSize_) / stride_) + 1;
                oh_ = floor(float(ih - kernelSize_) / stride_) + 1;
                break;
            case PaddingType::same:
                padding_ = ((iw-1)*stride_ + kernelSize_ - iw)*0.5;
                ow_ = int( (iw - kernelSize_ + (2*padding_)) / stride_ ) + 1;
                oh_ = int( (ih - kernelSize_ + (2*padding_)) / stride_ ) + 1;
                break;
            default:
                ow_ = floor(float(iw - kernelSize_) / stride_) + 1;
                oh_ = floor(float(ih - kernelSize_) / stride_) + 1;
                break;
        }

        pDst_ = new Tensor{batchSize_, oh_, ow_, oc_};
        pW_   = new Tensor{ic, kernelSize_, kernelSize_, oc_};
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

                im2col(src) =   | 0   1  3  4 |
                                | 1   2  4  5 |  Channel 0
                                | 3   4  6  7 |
                                | 4   5  7  8 |____________
                                | 9  10 12 13 |
                                | 10 11 13 14 |  Channel 1
                                | 12 13 15 16 |
                                | 13 14 16 17 |____________
                                | 18 19 21 22 |
                                | 19 20 22 23 |  Channel 2
                                | 21 22 24 25 |
                                | 22 23 25 26 |____________
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
                padding_, padding_,
                stride_, stride_,
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

                                | 0   1  3  4 |
                                | 1   2  4  5 |
                                | 3   4  6  7 |
                                | 4   5  7  8 |
                                | 9  10 12 13 |
                    | F0 |      | 10 11 13 14 |
                    | F1 |  X   | 12 13 15 16 |  =
                                | 13 14 16 17 |
                                | 18 19 21 22 |
                                | 19 20 22 23 |
                                | 21 22 24 25 |
                                | 22 23 25 26 |

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
}
