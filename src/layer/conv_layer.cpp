#include "layer/conv_layer.h"

namespace mkt {

    // Constructor
    ConvLayer::ConvLayer(
        Layer* prevLayer,
        std::string id,
        int fh,
        int fw,
        int fc,
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
        fh_{fh},
        fw_{fw},
        fc_{fc},
        stride_h_{stride_h},
        stride_w_{stride_w},
        pad_h_{pad_h},
        pad_w_{pad_w},
        padding_type_{paddingType},
        dilation_h_{1},
        dilation_w_{1},
        Layer(LayerType::Convolution, actType, weightInitType, biasInitType)
    {
        MKT_Assert(fc_ > 0, "fc_ = 0");
        MKT_Assert(fh_ > 0, "fh_ = 0");
        MKT_Assert(fw_ > 0, "fw_ = 0");
        MKT_Assert(stride_h_ > 0, "stride_h_ = 0");
        MKT_Assert(stride_w_ > 0, "stride_w_ = 0");


        id_ = id;
        batchSize_ = prevLayer->pDst_->getNumOfData();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();
        // int size3D = prevLayer->pDst_->getgetSize3D();

        pPrevLayer_ = prevLayer;

        oc_ = fc_;

        // Calculate oh, ow by input and filter dimension
        calcOutputSize(ic, ih, iw);

        pDst_ = new Tensor{batchSize_, oh_, ow_, oc_};
        pgDst_ = new Tensor{batchSize_, oh_, ow_, oc_};

        pW_   = new Tensor{ic, fh_, fw_, fc_};
        pgW_  = new Tensor{ic, fh_, fw_, fc_};

        pB_   = new Tensor{1, 1, 1, oc_};
        pgB_  = new Tensor{1, 1, 1, oc_};


        pTmpCol_ = new Tensor{1, pW_->getSize2D()*ic, pDst_->getSize2D(), 1};

        // Activator
        applyActivator();
    }

    // construct with LayerParams
    ConvLayer::ConvLayer(Layer* prevLayer, std::string id, LayerParams params):Layer(LayerType::Convolution)
    {
        id_ = id;

        batchSize_ = prevLayer->pDst_->getNumOfData();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

        pPrevLayer_ = prevLayer;

        // Parameter setting
        activationType_ = params.actType;
        weightInitType_ = params.weight_init_type;
        biasInitType_   = params.bias_init_type;

        fc_ = params.fc;
        fh_ = params.fh;
        fw_ = params.fw;

        stride_h_ = params.stride_h;
        stride_w_ = params.stride_w;

        padding_type_ = params.padding_type;
        pad_h_ = params.pad_h;
        pad_w_ = params.pad_w;

        // Temporary set dilation to 1
        dilation_h_ = params.dilation_h;
        dilation_w_ = params.dilation_w;

        oc_ = fc_;

        MKT_Assert(fc_ > 0, "fc_ = 0");
        MKT_Assert(fh_ > 0, "fh_ = 0");
        MKT_Assert(fw_ > 0, "fw_ = 0");
        MKT_Assert(stride_h_ > 0, "stride_h_ = 0");
        MKT_Assert(stride_w_ > 0, "stride_w_ = 0");

        // Calculate oh, ow by input and filter dimension
        calcOutputSize(ic, ih, iw);

        pDst_ = new Tensor{batchSize_, oh_, ow_, oc_};
        pgDst_ = new Tensor{batchSize_, oh_, ow_, oc_};

        pW_   = new Tensor{ic, fh_, fw_, fc_};
        pgW_  = new Tensor{ic, fh_, fw_, fc_};

        pB_   = new Tensor{1, 1, 1, oc_};
        pgB_  = new Tensor{1, 1, 1, oc_};


        pTmpCol_ = new Tensor{1, pW_->getSize2D()*ic, pDst_->getSize2D(), 1};

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

        initGradTensor();
        initGradWeightTensor();
        initGradBiasTensor();


        // temporary memory for im2col and col2im
        pTmpCol_->allocate();

    }

    // Computation Function
    void ConvLayer::Forward() {

        Tensor* pSrc = pPrevLayer_->pDst_;

        float* pSrcData = pSrc->getCPUData();
        float* pDstData = pDst_->getCPUData();
        float* pWData = pW_->getCPUData();
        float* pTmpColData = pTmpCol_->getCPUData();

        int ic = pSrc->getChannel();
        int iw = pSrc->getWidth();
        int ih = pSrc->getHeight();
        int src_size3D = pSrc->getSize3D();
        int src_wholeSize = pSrc->getWholeSize();

        int batchSize = pDst_->getNumOfData();
        int oc = pDst_->getChannel();
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
            Im2col
            For example:
                src     =ih(3) * iw(3) * ic(3)
                filter  = ic(3) * fh(2) * fw(2) * fc(2)
                padding = 0
                stride  = 1
                dst     = oh(2) * ow(2)

            src =   | 0 1 2 | |  9 10 11 | | 18 19 20 |
                    | 3 4 5 | | 12 13 14 | | 21 22 23 |
                    | 6 7 8 | | 15 16 17 | | 24 25 26 |

            According to the dimension of src and filter
            The filter is
            Filter_0    |w0_0_0, w0_0_1||w1_0_0, w1_0_1||w2_0_0, w2_0_1|
                        |w0_0_2, w0_0_3||w1_0_2, w1_0_3||w2_0_2, w2_0_3|

            Filter_1    |w0_1_0, w0_0_1||w1_1_0, w0_0_1||w2_1_0, w0_0_1|
                        |w0_1_2, w0_0_3||w1_1_2, w0_0_3||w2_1_2, w0_0_3|

            wi_f_c,
            i=index of scr channel
            f=index of filter
            c=index of weight of filter


            The first region of src which is applied to filter is
            |0 1| convert to col vector= |0|
            |3 4|                        |1|
                                         |3|
                                         |4|

            Conver src from image to column vector
            im2col(src) =   | s00 s01 s03 s04 |
                            | s01 s02 s04 s05 |  src Channel 0
                            | s03 s04 s06 s07 |
                            | s04 s05 s07 s08 |____________
                            | s09 s10 s12 s13 |
                            | s10 s11 s13 s14 |  src Channel 1
                            | s12 s13 s15 s16 |
                            | s13 s14 s16 s17 |____________
                            | s18 s19 s21 s22 |
                            | s19 s20 s22 s23 |  src Channel 2
                            | s21 s22 s24 s25 |
                            | s22 s23 s25 s26 |____________
                               |   |   |   |
                               |   |   |   |__________________
                               |   |   |____________          |
                               |   |_______         |         |
                               |           |        |         |
                        =   | patch_0 , patch_1, patch_2, patch_3 |

            im2col matrix = (ic*fh*fw) X (oh*ow)
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
                src     = ih(3) * iw(3) * ic(3)
                filter  = ( ic(3) * fh(2) * fw(2) ) * fc(2)
                padding = 0
                stride  = 1
                dst     = oh(2) X ow(2)

            1. Convert filter from |w0,w1| to [w0, w1, w2, w3]
                                   |w2,w3|

                According to the dimension of src and filter
                The filter is
                Filter_0    |w0_0_0,w0_0_1||w1_0_0,w1_0_1||w2_0_0,w2_0_1|
                            |w0_0_2,w0_0_3||w1_0_2,w1_0_3||w2_0_2,w2_0_3|

                Filter_1    |w0_1_0,w0_0_1||w1_1_0,w0_0_1||w2_1_0,w0_0_1|
                            |w0_1_2,w0_0_3||w1_1_2,w0_0_3||w2_1_2,w0_0_3|

                wi_f_c, i=index of scr channel
                        f=index of filter
                        c=index of weight of filter

                The filter matrix is
                | w0_0_0, ..., w0_0_3, w1_0_4, ..., w1_0_7, w2_0_8, ..., w2_0_11 | = filter 0 (F0)
                | w0_1_0, ..., w0_1_3, w1_1_4, ..., w1_1_7, w2_1_8, ..., w2_0_11 | = filter 1 (F1)

                W matrix = oc X (fh*fw*ic)


            2. Dst matrix = W X im2col(src) =

                            |  s0  s1  s3  s4 |
                            |  s1  s2  s4  s5 |
                            |  s3  s4  s6  s7 |
                            |  s4  s5  s7  s8 |
                            |  s9 s10 s12 s13 |
                | F0 |      | s10 s11 s13 s14 |
                | F1 |  X   | s12 s13 s15 s16 |
                            | s13 s14 s16 s17 |
                            | s18 s19 s21 s22 |
                            | s19 s20 s22 s23 |
                            | s21 s22 s24 s25 |
                            | s22 s23 s25 s26 |

                Dst matrix = oc X (oh*ow)

            */

            mkt::gemm_cpu(
                CblasNoTrans, CblasNoTrans,                          /* trans_A, trans_B*/
                pW_->getChannel(), pDst_->getSize2D(), pW_->getSize2D()*ic,   /* M,       N, K*/
                1.0f,                                                /* ALPHA */
                pWData, pW_->getSize2D()*ic,                            /* A,       lda(K)*/
                pTmpColData,   oh*ow,                                /* B,       ldb(N)*/
                1.0f,                                                /* BETA */
                pDstData, oh*ow                                      /* C,       ldc(N)*/
            );
        }


        // 2. Z = WX + Bias
        // addBias();

        // 3. A = activation(Z) = the input of next layer
        if (activationType_ != ActivationType::NONE)
        {
            pActivator_->Forward(*pDst_, *pDst_);
        }

    }
    void ConvLayer::Backward() {

        // backpropagation from dst.grad_data to src.grad_data
        // Step 1. dst_grad X weight
        /*

        1. w = [w0, w1, w2, w3]

        2. dst_grad = [d0, d1, d2, d3]

                 | w0 |
        3. w_t = | w1 |
                 | w2 |
                 | w3 |

                                 | w0 |                       | w0d0, w0d1, w0d2, w0d3 |   | c0 |
        4. gemm(w_t, dst_grad) = | w1 |  X [d0, d1, d2, d3] = | w1d0, w1d1, w1d2, w1d3 | = | c1 |
                                 | w2 |                       | w2d0, w2d1, w2d2, w2d3 |   | c2 |
                                 | w3 |                       | w3d0, w3d1, w3d2, w3d3 |   | c3 |


        base on reverse convolution, the src_grad_data is

                            | w0d0      , w0d1                , w1d1      |
        5. src_grad_data =  | w0d2+w2d0 , w0d3+w1d2+w2d1+w3d0 , w1d3+w3d1 |
                            | w2d2      , w2d3+w3d2           , w3d3      |

            c0m, c1m, c2m, c3m = convert c0, c1, c2, c3 from column vector(col) to matrix(im))

            c0m = c0( | w0d0, w0d1, w0d2, w0d3 | ) COL_TO_IM = |w0d0 , w0d1|
                                                               |w0d2 , w0d3|

            c1m = c0( | w1d0, w1d1, w1d2, w1d3 | ) COL_TO_IM = |w0d1 , w1d1|
                                                               |w1d2 , w1d3|

            c2m = c0( | w2d0, w2d1, w2d2, w2d3 | ) COL_TO_IM = |w2d0 , w2d1|
                                                               |w2d2 , w2d3|

            c3m = c0( | w3d0, w3d1, w3d2, w3d3 | ) COL_TO_IM = |w3d0 , w3d1|
                                                               |w3d2 , w3d3|

            src_grad_data is composited by c0m, c1m, c2m, c3m

            here display src_grad_data is composited by c0m, c1m, c2m, c3m
            part of c0m, c1m, c2m, c3m are overlaped.

            Left-Top of src_grad_data      Right-Top of src_grad_data
                | w0d0 , w0d1|                | w0d1 , w1d1|
                | w0d2 , w0d3|                | w1d2 , w1d3|

                | w2d0 , w2d1|                | w3d0 , w3d1|
                | w2d2 , w2d3|                | w3d2 , w3d3|
            Left-Bottom of src_grad_data      Right-bottom of src_grad_data

        */
        float* pWData = pW_->getCPUData();
        float* pgDstData = pgDst_->getCPUData();
        float* pTmpColData = pTmpCol_->getCPUData();

        int m = oc_; // num of filter
        int n = pW_->getSize2D() * pW_->getNumOfData();
        int k = pgDst_->getSize2D();
        mkt::gemm_cpu(
            CblasTrans, CblasNoTrans,               // trans_a, trans_b
            n, k, m,            // M, N, K
            1.0f,               // Alpha
            pWData, n,          // A,       lda
            pgDstData, k,       // B,       lda
            0,                  // Beta
            pTmpColData, k      // C,       lda
        );

        Tensor* pgSrc = pPrevLayer_->pgDst_;
        float* pgSrcData = pgSrc->getCPUData();
        int ic = pgSrc->getChannel();
        int ih = pgSrc->getHeight();
        int iw = pgSrc->getWidth();

        mkt::col2im_cpu(
            pTmpColData,
            ic, ih, iw,
            fh_, fw_,
            pad_h_, pad_w_,
            stride_h_, stride_w_,
            dilation_h_, dilation_w_,
            pgSrcData);
    }

    void ConvLayer::calcOutputSize(int ic, int ih, int iw) {
        switch(padding_type_) {
            case PaddingType::VALID:
                ow_ = static_cast<int>( static_cast<float>(iw - fw_ + 2*pad_w_) / stride_w_) + 1;
                oh_ = static_cast<int>( static_cast<float>(ih - fh_ + 2*pad_h_) / stride_h_) + 1;
                break;
            case PaddingType::SAME:

                pad_w_ = static_cast<int>( ((iw-1)*stride_w_ + fw_ - iw) * 0.5 );
                ow_    = static_cast<int>( static_cast<float>(iw - fw_ + (2*pad_w_)) / stride_w_ )  + 1;
                MKT_Assert(iw == ow_, "iw != ow");

                pad_h_ = static_cast<int>( ((ih-1)*stride_h_ + fh_ - ih) * 0.5 );
                oh_    = static_cast<int>( static_cast<float>(ih - fh_ + (2*pad_h_)) / stride_h_ ) + 1;
                MKT_Assert(ih == oh_, "ih != oh");

                break;
            default:
                ow_ = static_cast<int>( static_cast<float>(iw - fw_) / stride_w_)  + 1;
                oh_ = static_cast<int>( static_cast<float>(ih - fh_) / stride_h_)  + 1;
                break;
        }
    }

    // Getter
    int ConvLayer::getFiltergetHeight() {
        return fh_;
    }
    int ConvLayer::getFiltergetWidth() {
        return fw_;
    }
    int ConvLayer::getFiltergetChannel() {
        return fc_;
    }
}
