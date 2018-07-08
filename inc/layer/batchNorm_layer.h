
/**
 * Implement Batch normalization which is proposed in the paper
 * "Batch Normalization: Accelerating Deep Network Training by
 *  Reducing Internal Covariate Shift". S. Ioffe and C. Szegedy (2015)
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

const double DEFAULT_BATCH_NORM_EPS = 0.0001;

template<typename T>
class BatchNormLayer: public Layer<T>
{
public:
    BatchNormLayer(Layer<T>* prevLayer, std::string id, InitializerType weightInitType,
        InitializerType biasInitType, T eps=DEFAULT_BATCH_NORM_EPS);
    ~BatchNormLayer();

    // Must Implement virtual finctions form Layer class
    void initialize(NetMode mode);
    void Forward();
    void Backward();
    void serialize(std::fstream& fileHandler, bool bWriteInfo);
    void deserialize(std::fstream& fileHandler, bool bWriteInfo);

private:
    T eps_;
    int running_stats_window_size_;
    int num_updates_;
    Tensor<T>* pRunning_means_;     // for inference
    Tensor<T>* pRunning_variances_; // for inference

    // Temporary use
    Tensor<T>* pMean_;              // current batch data mean
    Tensor<T>* pInvstds_;           // current batch data variance = sigmae square
    Tensor<T>* pdvar_;
    Tensor<T>* pdmean_;

    // Here i use pW_ to be "gamma" and pB_ to be "beta"
    // pW_  = gamma = learned scale parameter
    // pB_  = beta  = learned shift parameter
    // pgW_ = gradient of gamma
    // pgB_ = gradient of beta

};

} // namespace mkt

#endif //MKT_BATCHNORM_H
