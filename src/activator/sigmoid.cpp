
#include "activator/sigmoid_act.h"

namespace mkt {
    inline float sigmoid(float x) {
        return ( 1.0f / (1.0f + exp(-x)) );
    }


    // Constructor
    template<typename T>
    Sigmoid_Act<T>::Sigmoid_Act() {};

    // Destructor
    template<typename T>
    Sigmoid_Act<T>::~Sigmoid_Act(){};

    template<typename T>
    void Sigmoid_Act<T>::Forward(Tensor<T> *src, Tensor<T> *dst) {

        // Check size
        int srcWholeSize = src->getWholeSize();
        int dstWholeSize = src->getWholeSize();
        CHECK_EQ(srcWholeSize, dstWholeSize, __func__);
        // if (!Check_EQ(srcWholeSize, dstWholeSize))
        // {
        //     MKT_ERR_LOG("Relu:Forward, size of src(%d) and dst(%d) is not equal\n", srcWholeSize, dstWholeSize);
        //     return;
        // }

        // Get data memory
        T* pSrcData = src->getCPUData();
        T* pDstData = dst->getCPUData();

        for (int i = 0; i < dstWholeSize; ++i)
        {
            pDstData[i] = sigmoid(pSrcData[i]);
        }
    };

    template<typename T>
    void Sigmoid_Act<T>::Backward(Tensor<T> *src, Tensor<T> *src_grad, Tensor<T> *dst_grad) {
        fprintf(stderr, "%s, %s, %d not yet implement\n", __FILE__, __func__, __LINE__);
    };

    // Explicitly instantiate the template, and its member definitions
    template class Sigmoid_Act<float>;

} // namespace mkt


