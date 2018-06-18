

#include "operations/softmax_operations.h"


namespace mkt {
namespace op {

    template<typename T>
    void softmax(const int num_samples, Tensor<T>* pSrc, Tensor<T>* pDst)
    {
        // Note that i use the channel as softmax axis

        const auto pSrcData = pSrc->getCPUData();
        int ic = pSrc->getChannel();
        int size2D = pSrc->getSize2D();
        const auto pDstData = pDst->getCPUData();

        // Note that we subtract out the max values in each channel before
        // applying exp() to avoid numeric overflow in the subsequent
        // computations. Doing this doesn't change the resulting output, it
        // just makes it more numerically stable.
        for (size_t b = 0; b < num_samples; ++b)
        {
            auto curSrcData = pSrcData + b * ic * size2D;
            auto curDstData = pDstData + b * ic * size2D;

            for (size_t sz = 0; sz < size2D; ++sz)
            {
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t c = 0; c < ic; ++c)
                {
                    max_val = std::max(max_val, curSrcData[c*size2D]);
                }
                for (size_t c = 0; c < ic; ++c)
                {
                    curDstData[c*size2D] = exp(curSrcData[c*size2D]-max_val);
                }

                ++curSrcData;
                ++curDstData;
            }
        }

        // Normalize each channel so tyeh sum to 1.
        for (size_t b = 0; b < num_samples; ++b)
        {
            const auto curDstData = pDstData + b * ic * size2D;
            for (size_t sz = 0; sz < size2D; ++sz)
            {
                const auto pTemp = curDstData + sz;
                float temp = 0;
                for (size_t c = 0; c < ic; ++c)
                {
                    temp += pTemp[c*size2D];
                }
                for (size_t c = 0; c < ic; ++c)
                {
                    pTemp[c*size2D] /= temp;
                }
            }
        }
    }

    template<typename T>
    void softmax_gradient(const int num_samples, Tensor<T>* pgSrc, Tensor<T>* pDst, Tensor<T>* pgDst)
    {
        const auto pDstData = pDst->getCPUData();
        const auto pgDstData = pgDst->getCPUData();
        const auto pgSrcData = pgSrc->getCPUData();

        int size2D = pDst->getSize2D();
        int oc = pDst->getChannel();

        for (int b = 0; b < num_samples; ++b)
        {
            const auto pDst2 = pDstData + b * size2D * oc;
            const auto pgDst2 = pgDstData + b * size2D * oc;
            const auto pgSrc2 = pgSrcData + b * size2D * oc;

            for (int sz = 0; sz < size2D; ++sz)
            {
                const auto pDst3 = pDst2 + sz;
                const auto pgDst3 = pgDst2 + sz;
                const auto pgSrc3 = pgSrc2 + sz;

                float temp = 0;
                for (int c = 0; c < oc; ++c)
                {
                    temp += -pDst3[c*size2D] * pgDst3[c*size2D];
                }
                for (int c = 0; c < oc; ++c)
                {
                    pgSrc3[c*size2D] + pDst3[c*size2D]*(temp+pgDst3[c*size2D]);
                }
            }
        }
    }


    // Explicit instantiation
    template void softmax<float>(const int, Tensor<float>*, Tensor<float>*);

    template void softmax_gradient<float>(const int, Tensor<float>*, Tensor<float>*, Tensor<float>*);


} // namespace op
} // namespace mkt
