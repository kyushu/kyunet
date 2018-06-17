
#include "layer/batchNorm_layer.h"
#include "operations/bn_operations.h"

namespace mkt {

    template<typename T>
    BatchNormLayer<T>::BatchNormLayer(Layer<T>* prevLayer, std::string id, InitializerType weightInitType,
        InitializerType biasInitType, T eps):eps_{eps}, num_updates_{0}, Layer<T>(LayerType::BATCHNORM)
    {

        this->id_ = id;
        this->batchSize_ = prevLayer->pDst_->getNumOfData();
        this->pPrevLayer_ = prevLayer;

        this->oc_ = this->pPrevLayer_->pDst_->getChannel();
        this->oh_ = this->pPrevLayer_->pDst_->getHeight();
        this->ow_ = this->pPrevLayer_->pDst_->getWidth();


        // The dimemsion of output is same as previous output
        this->pDst_ = new Tensor<T>{this->batchSize_, this->oh_, this->ow_, this->oc_};
        fprintf(stderr, "b: %d, oh: %d, ow: %d, oc: %d\n", this->batchSize_, this->oh_, this->ow_, this->oc_);
        // pW_ = gamma
        this->pW_   = new Tensor<T>{1, 1, 1, this->oc_};
        // pB_ = beta
        this->pB_   = new Tensor<T>{1, 1, 1, this->oc_};


        this->pgDst_ = new Tensor<T>{this->batchSize_, this->oh_, this->ow_, this->oc_};
        this->pgW_   = new Tensor<T>{1, 1, 1, this->oc_};
        this->pgB_   = new Tensor<T>{1, 1, 1, this->oc_};

        // running mean and variance are used to compute
        // x_norm when inference
        pRunning_means_     = new Tensor<T>{1, 1, 1, this->oc_};
        pRunning_variances_ = new Tensor<T>{1, 1, 1, this->oc_};

        // For current batch data
        pMean_              = new Tensor<T>{1, 1, 1, this->oc_};
        pInvstds_           = new Tensor<T>{1, 1, 1, this->oc_};
        pdmean_             = new Tensor<T>{1, 1, 1, this->oc_};
        pdvar_              = new Tensor<T>{1, 1, 1, this->oc_};



        // Set initial value of gamma = 1
        this->weightInitType_ = weightInitType;
        // Set initial value of beta = 0
        this->biasInitType_ = biasInitType;
    }

    template<typename T>
    BatchNormLayer<T>::~BatchNormLayer()
    {
        delete pRunning_means_;
        delete pRunning_variances_;
        delete pMean_;
        delete pInvstds_;
        delete pdmean_;
        delete pdvar_;
    }

    template<typename T>
    void BatchNormLayer<T>::initialize(NetMode mode)
    {
        MKT_Assert(this->pDst_  != nullptr, "pDst_ is null");
        MKT_Assert(this->pW_    != nullptr, "pW_ is null");
        MKT_Assert(this->pB_    != nullptr, "pB_ is null");

        this->initOutputTensor();
        this->initWeightTensor();
        this->initBiasTensor();

        pRunning_means_->allocate();
        std::fill_n(pRunning_means_->getCPUData(), pRunning_means_->getWholeSize(), 0);
        pRunning_variances_->allocate();
        std::fill_n(pRunning_variances_->getCPUData(), pRunning_variances_->getWholeSize(), 0);


        MKT_Assert(this->pgDst_ != nullptr, "pgDst_ is null");
        MKT_Assert(this->pgW_   != nullptr, "pgW_ is null");
        MKT_Assert(this->pgB_   != nullptr, "pgB_ is null");

        this->initGradTensor();
        this->initGradWeightTensor();
        this->initGradBiasTensor();

        pMean_->allocate();
        // std::fill_n(pMu_->getCPUData(), pMu_->getWholeSize(), 0);
        pInvstds_->allocate();
        // std::fill_n(pVariance_->getCPUData(), pVariance_->getWholeSize(), 0);

        pdmean_->allocate();
        pdvar_->allocate();

    }

    template<typename T>
    void BatchNormLayer<T>::Forward()
    {
        // Reset data
        this->pDst_->resetData();
        this->pgDst_->resetData();
        this->pgW_->resetData();
        this->pgB_->resetData();

        pMean_->resetData();
        pInvstds_->resetData();
        pdvar_->resetData();
        pdmean_->resetData();

        // averaging_factor is used for running mean and variance
        const T averaging_factor = 1.0 - num_updates_/(num_updates_ + 1.0);
        ++num_updates_;
        if (num_updates_ > running_stats_window_size_) {
            num_updates_ = running_stats_window_size_;
        }

        op::bn::batchNorm (
            this->batchSize_, averaging_factor, eps_,
            this->pPrevLayer_->pDst_, this->pDst_,
            this->pW_,  this->pB_,
            pMean_,  pInvstds_, pRunning_variances_, pRunning_means_
        );

    }

    template<typename T>
    void BatchNormLayer<T>::Backward()
    {
        op::bn::gradientBatchNorm (
        this->batchSize_,
        this->pPrevLayer_->pDst_, this->pPrevLayer_->pgDst_, this->pgDst_,
        this->pW_, this->pgW_, this->pgB_,
        pMean_, pInvstds_,
        pdmean_, pdvar_);

        // int channel = this->pPrevLayer_->pDst_->getChannel();
        // int size2D = this->pPrevLayer_->pDst_->getSize2D();

        // auto pSrcData = this->pPrevLayer_->pDst_->getCPUData();
        // auto pWData = this->pW_->getCPUData();
        // auto pMeanData = pMean_->getCPUData();
        // auto pInvstdsData = pInvstds_->getCPUData();

        // auto pgDstData = this->pgDst_->getCPUData();
        // auto pgWData = this->pgW_->getCPUData();
        // auto pgBData = this->pgB_->getCPUData();

        // auto pgVarsData = pdvar_->getCPUData();
        // auto pgMeansData = pdmean_->getCPUData();

        // // Update gradient with respect to gamma(pW_), beta(pB_), variance
        // for (size_t b = 0; b < this->batchSize_; ++b)
        // {
        //     for (size_t c = 0; c < channel; ++c)
        //     {
        //         const T invstd_pow = -0.5*std::pow(pInvstdsData[c], 3.0f);
        //         for (size_t sz = 0; sz < size2D; ++sz)
        //         {
        //             const T x_hat = (*pSrcData - pMeanData[c])*pInvstdsData[c];
        //             pgBData[c] += *pgDstData;
        //             pgWData[c] += (*pgDstData)*x_hat;

        //             const T dx = *pgDstData * pgWData[c];

        //             pgVarsData[c] += dx*(*pSrcData - pMeanData[c])*invstd_pow;

        //             ++pgDstData;
        //             ++pSrcData;
        //         }
        //     }
        // }

        // // Update gradient with respect to mean
        // pgDstData = this->pgDst_->getCPUData();
        // pSrcData = this->pPrevLayer_->pDst_->getCPUData();
        // const float invnum = 1.0f / size2D * this->batchSize_;
        // for (size_t b = 0; b < this->batchSize_; ++b)
        // {
        //     for (size_t c = 0; c < channel; ++c)
        //     {
        //         for (size_t sz = 0; sz < size2D; ++sz)
        //         {
        //             const float dx = *pgDstData * pgWData[c];

        //             pgMeansData[c] += -dx*pInvstdsData[c] + pgVarsData[c] * -2*(*pSrcData - pMeanData[c])*invnum;

        //             ++pgDstData;
        //             ++pSrcData;
        //         }
        //     }
        // }

        // auto pgSrcData = this->pPrevLayer_->pgDst_->getCPUData();
        // pSrcData = this->pPrevLayer_->pDst_->getCPUData();
        // pgDstData = this->pgDst_->getCPUData();
        // for (size_t b = 0; b < this->batchSize_; ++b)
        // {
        //     for (size_t c = 0; c < channel; ++c)
        //     {
        //         for (size_t sz = 0; sz < size2D; ++sz)
        //         {
        //             const T dx = *pgDstData * pWData[c];

        //             *pgSrcData += dx*pInvstdsData[c] + pgVarsData[c]*2*(*pSrcData - pMeanData[c])*invnum + pgMeansData[c]*invnum;

        //             ++pgDstData;
        //             ++pSrcData;
        //             ++pgSrcData;

        //         }
        //     }
        // }
    }


    template class BatchNormLayer<float>;
} // namespace mkt


/**
 *
 */
