
/**
 * Implement Batch normalization which is proposed in the paper
 * "Batch Normalization: Accelerating Deep Network Training by
 *  Reducing Internal Covariate Shift"
 *
 * The purpose of Batch normalization is trying to reduce "Internal
 *  Covariant Shift" Which is "the change in the distribution of network
 *  activations due to the change in network parameters during training".
 *
 * According to the paper, by fixing the distribution of the layer inputs x
 *  as the training progresses, we expect to improve the training speed.
 *
 * For fully-Connected layer, the dimemsion of Gamma and Beta is MxC,
 * M is batch size and C is number of hidden node of layer.
 * For Convolution layer, the dimension of Gamma and Beta is MXC,
 * M is batch size and C is number of kernel(filter).
 */

#ifndef MKT_BATCHNORM_H
#define MKT_BATCHNORM_H

#include "layer.h"

namespace mkt {

template<typename T>
class BatchNorm: public Layer<T>
{
public:
    BatchNorm(Layer<T>* prevLayer, std::string id);
    ~BatchNorm();

    // Must Implement virtual finctions form Layer class
    void initialize(NetMode mode);
    void Forward();
    void Backward();

private:
    T mu;             // mean
    T variance;       // variance = sigmae square

    // Here i use pW_ to be "gamma" and pB_ to be "beta"
    // pW_  = gamma = learned scale parameter
    // pB_  = beta  = learned shift parameter
    // pgW_ = gradient of gamma
    // pgB_ = gradient of beta

};

} // namespace mkt

#endif //MKT_BATCHNORM_H
