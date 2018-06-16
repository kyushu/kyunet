#include "activator/relu_act.h"

namespace mkt {

    // Constructor
    template<typename T>
    Relu_Act<T>::Relu_Act():negative_slope_{0} {};

    // Destructor
    template<typename T>
    Relu_Act<T>::~Relu_Act() {};

    template<typename T>
    void Relu_Act<T>::Forward(Tensor<T> *src, Tensor<T> *dst) {

        // Check size
        int srcWholeSize = src->getWholeSize();
        int dstWholeSize = src->getWholeSize();
        CHECK_EQ(srcWholeSize, dstWholeSize, __func__, __LINE__);
        // if (!Check_EQ(srcWholeSize, dstWholeSize))
        // {
        //     MKT_ERR_LOG("Relu:Forward, size of src(%d) and dst(%d) is not equal\n", srcWholeSize, dstWholeSize);
        //     return;
        // }

        // Get data memory
        T* pSrcData = src->getCPUData();
        T* pDstData = dst->getCPUData();

        //
        for (int i = 0; i < dstWholeSize; ++i)
        {
            // negative_slop is used for leaky_relu
            pDstData[i] = std::max(pSrcData[i], 0.0f) + negative_slope_ * std::min(pSrcData[i], 0.0f);
        }

    };

    template<typename T>
    void Relu_Act<T>::Backward(Tensor<T> *src, Tensor<T> *src_grad, Tensor<T> *dst_grad) {

        T* pSrcData = src->getCPUData();
        T* pSrcGradData = src_grad->getCPUData();
        T* pDstGradData = dst_grad->getCPUData();

        int wholeSize = src->getWholeSize();
        for (int i = 0; i < wholeSize; ++i)
        {
            pSrcGradData[i] = pDstGradData[i] *
                            ( (pSrcData[i] > 0) + negative_slope_ * (pSrcData[i] <=0) );
        }
    };

    // Explicitly instantiate the template, and its member definitions
    template class Relu_Act<float>;

} // namespace mkt
