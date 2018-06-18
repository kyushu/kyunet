
#ifndef SOFT_OPERATIONS_H
#define SOFT_OPERATIONS_H

#include <math.h>       /* exp */
#include <algorithm>    /* std::max */
#include <limits>
#include "tensor.h"

namespace mkt  {
namespace op   {

    template<typename T>
    void softmax(const int num_samples, Tensor<T>* pSrc, Tensor<T>* pDst);

    template<typename T>
    void softmax_gradient(const int num_samples, Tensor<T>* pgSrc, Tensor<T>* pDst, Tensor<T>* pgDst);


} // namespace op
} // namespace mkt

#endif
