#include "convLayer.h"

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

        int batchSize = prevLayer->pDst_->getNumOfData();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getDepth();
        int size3D = prevLayer->pDst_->getSize3D();

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

        pDst_ = new Tensor{batchSize, oh_, ow_, oc_};
        pW_   = new Tensor{ic, kernelSize_, kernelSize_, oc_};
        pB_   = new Tensor{1, 1, 1, oc_};

        pTmpCol_ = new Tensor{1, pW_->getSize2D()*ic, pDst_->getSize2D(), oc_};

        // TODO: Activation setting
    }

    // Destructor
    ConvLayer::~ConvLayer() {

        delete pTmpCol_;
    }


    // Initialization
    void ConvLayer::initialize() {

        initOutputTensor();
        initWeightTensor(weightInitType_);
        initBiasTensor(biasInitType_);

        // temporary memory for im2col and col2im
        pTmpCol_->initialize(InitializerType::NONE);

    }

    // Computation Function
    void ConvLayer::forward() {
        int ic = pSrc_->getDepth();
        int batchSize = pDst_->getNumOfData();
        int oh = pDst_->getHeight();
        int ow = pDst_->getWidth();
        for (int i = 0; i < batchSize; ++i)
        {

            // im2col
            mkt::im2col_cpu(pSrc_->pData_,
                pSrc_->getDepth(), pSrc_->getHeight(), pSrc_->getWidth(),
                pW_->getHeight(), pW_->getWidth(),
                padding_, padding_,
                stride_, stride_,
                dilation_h_, dilation_w_,
                pTmpCol_->pData_
            );


            // GEMM: kernel X im2col
            mkt::gemm_cpu(
                0, 0,                                                           /*trans_A, trans_B*/
                pW_->getNumOfData(), pDst_->getSize2D(), pW_->getSize2D()*ic,   /*M,       N, K*/
                1.0f, 1.0f,                                                     /*ALPHA,   BETA*/
                pW_->pData_, pW_->getSize2D()*ic,                               /*A,       lda(K)*/
                pTmpCol_->pData_,   oh*ow,                                      /*B,       ldb(N)*/
                pDst_->pData_, oh*ow                                            /*C,       ldc(N)*/
            );
        }

    }
    void ConvLayer::backward() {

    }
}
