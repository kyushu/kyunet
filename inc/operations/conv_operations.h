
#ifndef CONV_OPERATIONS_H
#define CONV_OPERATIONS_H

#include "definitions.h"
#include "params.h"
#include "operations/mat_operations.h"
#include "tensor.h"
#include "params.h"

namespace mkt  {
namespace op   {
namespace conv {

    void calcOutputSize (
        const int& inputW, const int& inputH,
        const int& filterW, const int& filterH,
        ConvParam& param, int& ootputW, int& outputH );


    template<typename T>
    void convOperation (
        int numOfSample,  const ConvParam& convParam,
        Tensor<T>* pSrc, Tensor<T>* pDst,
        Tensor<T>* pW, Tensor<T>* pB,
        Tensor<T>* pTmpCol);

    template<typename T>
    void convGradient (
        int numOfSample, const ConvParam& convParam, const Shape& filterShape,
        Tensor<T>* pSrc, Tensor<T>* pgSrc, Tensor<T>* pgDst,
        Tensor<T>* pW, Tensor<T>* pgW, Tensor<T>* pgB,
        Tensor<T>* pTmpCol);

} // namespace conv
} // namespace op
} // namespace mkt

#endif
