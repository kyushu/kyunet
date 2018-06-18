#ifndef POOLING_OPERATIONS_H
#define POOLING_OPERATIONS_H

#include "tensor.h"
#include "params.h"

namespace mkt {
namespace op {

    template<typename T>
    void pooling (
        const int numOfSample, const PoolingMethodType poolingType, const ConvParam& convParam,
        const int fh, const int fw,
        Tensor<T>* pSrc, Tensor<T>* pDst, Tensor<T>* pMask);

    template<typename T>
    void pooling_gradient(const int num_sample, const PoolingMethodType poolingType, const ConvParam& convParam,
        const int fh, const int fw,
        Tensor<T>* pgSrc, Tensor<T>* pgDst , Tensor<T>* pMaxMask);

} // namespace op
} // namespace mkt

#endif
