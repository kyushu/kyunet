
#ifndef BN_OPERATIONS_H
#define BN_OPERATIONS_H

#include "definitions.h"
#include "params.h"
#include "operations/mat_operations.h"
#include "tensor.h"
#include "params.h"

namespace mkt  {
namespace op   {

    template <typename T>
    void batchNorm (
       const int numOfSample, const T averaging_factor, const T eps,
        Tensor<T>* pSrc, Tensor<T>* pDst,
        Tensor<T>* pW, Tensor<T>* pB,
        Tensor<T>* pMean, Tensor<T>* pInvstds,
        Tensor<T>* pRunning_variances, Tensor<T>* pRunning_means);

    template<typename T>
    void batchNorm_infer (
        int numOfSample, T eps,
        Tensor<T>* pSrc, Tensor<T>* pDst,
        Tensor<T>* pRunning_mean, Tensor<T>* pRunning_variances,
        Tensor<T>* pGamma, Tensor<T>* pBeta);


    template<typename T>
    void batchNorm_gradient (
        const int numOfSample,
        Tensor<T>* pSrc, Tensor<T>* pgSrc, Tensor<T>* pgDst,
        Tensor<T>* pW, Tensor<T>* pgW, Tensor<T>* pgB,
        Tensor<T>* pMean, Tensor<T>* pInvstds,
        Tensor<T>* pdmean, Tensor<T>* pdvar);


} // namespace op
} // namespace mkt

#endif
