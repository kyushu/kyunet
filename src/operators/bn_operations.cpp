
#include "operations/bn_operations.h"

namespace mkt {
namespace op {

    template <typename T>
    void batchNorm (std::string id,
        const int numOfSample, const T averaging_factor, const T eps,
        Tensor<T>* pSrc, Tensor<T>* pDst,
        Tensor<T>* pW, Tensor<T>* pB,
        Tensor<T>* pMean, Tensor<T>* pInvstds,
        Tensor<T>* pRunning_variances, Tensor<T>* pRunning_means)
    {

        auto pSrcData = pSrc->getCPUData();
        auto pMeanData = pMean->getCPUData();
        auto pInvstdsData = pInvstds->getCPUData();
        auto pDstData = pDst->getCPUData();

        auto pWData = pW->getCPUData();
        auto pBData = pB->getCPUData();

        int channel = pSrc->getChannel();
        int size2D = pSrc->getSize2D();

        if (id == "bn4")
        {
            fprintf(stderr, "%s forward\n", id.c_str());
        }
        // Compute mean and invert variance
        for (size_t b = 0; b < numOfSample; ++b)
        {
            for (size_t c = 0; c < channel; ++c)
            {
                for (size_t sz = 0; sz < size2D; ++sz)
                {

                    // if (id == "bn4")
                    // {
                    //     // fprintf(stderr, "%s forward\n", id.c_str());
                    //     fprintf(stderr, "pSrcData=%f\t", c, *pSrcData);
                    // }

                    pMeanData[c] += *pSrcData / (size2D*numOfSample);
                    pInvstdsData[c] += (*pSrcData)*(*pSrcData) / (size2D*numOfSample);
                    ++pSrcData;


                    
                }
            }
        }

        
        // Update running variance and invert_variance
        auto rvar = pRunning_variances->getCPUData();
        const double scale = (size2D*numOfSample) / (size2D*numOfSample - 1.0);
        for (int c = 0; c < channel; ++c)
        {
            T actual_var = pInvstdsData[c] - pMeanData[c]*pMeanData[c];

            
            if (averaging_factor == 1)
            {
                rvar[c] = scale*actual_var;
            }
            else {
                rvar[c] = (1-averaging_factor)*rvar[c] + scale*averaging_factor*actual_var;
            }
            pInvstdsData[c] = 1.0/std::sqrt(actual_var + eps);
        }

        // if (id == "bn4")
        // {
        //     fprintf(stderr, "%s forward\n", id.c_str());
        //     for (int i = 0; i < pInvstds->getSize3D(); ++i)
        //     {
        //         fprintf(stderr, "invstd=%f\n", pInvstdsData[i]);
        //     }
        // }

        pSrcData = pSrc->getCPUData();
        // x_norm = (x-m)/variance
        // y = gamma*x_norm - beta
        for (size_t b = 0; b < numOfSample; ++b)
        {
            // fprintf(stderr, "b: %d\n", b);
            for (size_t c = 0; c < channel; ++c)
            {
                for (size_t sz = 0; sz < size2D; ++sz)
                {
                    *pDstData = (*pSrcData - pMeanData[c])*pInvstdsData[c];
                    *pDstData = (*pDstData)*pWData[c] + pBData[c];

                    ++pSrcData;
                    ++pDstData;

                }
            }
        }

        // Update running mean
        auto rmean = pRunning_means->getCPUData();
        if (averaging_factor != 1)
        {
            // running_mean = (1-averaging_faceor)*running_mean + averaging_factor*mean
            op::mat::axpby(channel, averaging_factor, pMeanData, (1-averaging_factor), rmean);
        }
        else {
            // running_mean = mean
            op::mat::axpby(channel, 1, pMeanData, 0, rmean);
        }
    }

    template<typename T>
    void batchNorm_gradient (std::string id,
        const int numOfSample,
        Tensor<T>* pSrc, Tensor<T>* pgSrc, Tensor<T>* pgDst,
        Tensor<T>* pW, Tensor<T>* pgW, Tensor<T>* pgB,
        Tensor<T>* pMean, Tensor<T>* pInvstds,
        Tensor<T>* pdmean, Tensor<T>* pdvar)
    {

        auto pSrcData = pSrc->getCPUData();

        int channel = pSrc->getChannel();
        int size2D = pSrc->getSize2D();

        auto pWData = pW->getCPUData();
        auto pMeanData = pMean->getCPUData();
        auto pInvstdsData = pInvstds->getCPUData();

        auto pgDstData = pgDst->getCPUData();
        auto pgWData = pgW->getCPUData();
        auto pgBData = pgB->getCPUData();

        auto pgVarsData = pdvar->getCPUData();
        auto pgMeansData = pdmean->getCPUData();


        // TEST
        // fprintf(stderr, "bn_bw: %s\n", id.c_str());
        // for (int i = 0; i < pInvstds->getWholeSize(); ++i)
        // {
        //     fprintf(stderr, "inv_std[%d] = %f\n", i, pInvstdsData[i]);
        // }
        // TEST

        // Update gradient with respect to gamma(pW_), beta(pB_), variance
        for (size_t b = 0; b < numOfSample; ++b)
        {
            for (size_t c = 0; c < channel; ++c)
            {
                const T invstd_pow = -0.5*std::pow(pInvstdsData[c], 3.0f);
                for (size_t sz = 0; sz < size2D; ++sz)
                {
                    const T x_hat = (*pSrcData - pMeanData[c])*pInvstdsData[c];
                    pgBData[c] += *pgDstData;
                    pgWData[c] += (*pgDstData)*x_hat;

                    const T dx = *pgDstData * pgWData[c];

                    pgVarsData[c] += dx*(*pSrcData - pMeanData[c])*invstd_pow;

                    ++pgDstData;
                    ++pSrcData;
                }
            }
        }

        // Update gradient with respect to mean
        pgDstData = pgDst->getCPUData();
        pSrcData = pSrc->getCPUData();
        const float invnum = 1.0f / size2D * numOfSample;
        if (id == "bn4")
        {
            fprintf(stderr, "%s backward\n", id.c_str());
            fprintf(stderr, "invnum: %f\n", invnum);
        }
        
        for (size_t b = 0; b < numOfSample; ++b)
        {
            for (size_t c = 0; c < channel; ++c)
            {
                for (size_t sz = 0; sz < size2D; ++sz)
                {
                    const float dx = *pgDstData * pgWData[c];

                    pgMeansData[c] += -dx*pInvstdsData[c] + pgVarsData[c] * -2*(*pSrcData - pMeanData[c])*invnum;

                    ++pgDstData;
                    ++pSrcData;
                }
            }
        }

        // Update gradient with respect to data (x)
        auto pgSrcData = pgSrc->getCPUData();
        pSrcData = pSrc->getCPUData();
        pgDstData = pgDst->getCPUData();
        for (size_t b = 0; b < numOfSample; ++b)
        {
            for (size_t c = 0; c < channel; ++c)
            {
                for (size_t sz = 0; sz < size2D; ++sz)
                {
                    const T dx = *pgDstData * pWData[c];

                    *pgSrcData += dx*pInvstdsData[c] + pgVarsData[c]*2*(*pSrcData - pMeanData[c])*invnum + pgMeansData[c]*invnum;

                    ++pgDstData;
                    ++pSrcData;
                    ++pgSrcData;

                    // if (id == "bn4")
                    // {

                    //     if (b == 0 && c == 0)
                    //     {
                    //         fprintf(stderr, "dx: %f\t", dx);
                    //         fprintf(stderr, "invstd[%d]: %f\t", c, pInvstdsData[c]);
                    //         fprintf(stderr, "pgVarsData[%d]:%f\t", c, pgVarsData[c]);
                    //         fprintf(stderr, "pMeanData[%d]:%f\n", c, pMeanData[c]);

                    //     }
                    // }
                    
                }
            }
        }
    }


    template <typename T>
    void batchNorm_infer(
        int numOfSample, T eps,
        Tensor<T>* pSrc, Tensor<T>* pDst,
        Tensor<T>* pRunning_mean, Tensor<T>* pRunning_variances,
        Tensor<T>* pGamma, Tensor<T>* pBeta)
    {


        auto pSrcData = pSrc->getCPUData();
        auto pDstData = pDst->getCPUData();
        auto rmeans = pRunning_mean->getCPUData();
        auto rvars = pRunning_variances->getCPUData();
        auto pGammaData = pGamma->getCPUData();
        auto pBetaData = pBeta->getCPUData();

        const int channel = pSrc->getChannel();
        const int size2D = pSrc->getSize2D();
        for (size_t b = 0; b < numOfSample; ++b)
        {
            for (int c = 0; c < channel; ++c)
            {
                for (size_t sz = 0; sz < size2D; ++sz)
                {
                    *pDstData = pGammaData[c]*(*pSrcData - rmeans[c]) / std::sqrt(rvars[c] + eps) + pBetaData[c];

                    ++pDstData;
                    ++pSrcData;
                }
            }

        }
    }

    /**
     * Explicit instantiation
     */
    template void batchNorm<float>(std::string id,
        const int, const float , const float,
        Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*);

    template void batchNorm_gradient<float>(std::string id,
        const int,
        Tensor<float>*, Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*);

    template void batchNorm_infer<float>(
        int, float,
        Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*);


} // namespace op
} // namespace mkt
