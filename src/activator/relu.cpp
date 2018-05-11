#include "activator/relu_act.h"

namespace mkt {

    // Constructor
    Relu_Act::Relu_Act():negative_slope_{0} {};

    // Destructor
    Relu_Act::~Relu_Act() {};

    void Relu_Act::Forward(Tensor &src, Tensor &dst) {

        // Check size
        int srcWholeSize = src.getWholeSize();
        int dstWholeSize = src.getWholeSize();
        CHECK_EQ(srcWholeSize, dstWholeSize, __func__);
        // if (!Check_EQ(srcWholeSize, dstWholeSize))
        // {
        //     MKT_ERR_LOG("Relu:Forward, size of src(%d) and dst(%d) is not equal\n", srcWholeSize, dstWholeSize);
        //     return;
        // }

        // Get data memory
        float* pSrcData = src.getCPUData();
        float* pDstData = dst.getCPUData();

        //
        for (int i = 0; i < dstWholeSize; ++i)
        {
            // negative_slop is used for leaky_relu
            pDstData[i] = std::max(pSrcData[i], 0.0f) + negative_slope_ * std::min(pSrcData[i], 0.0f);
        }

    };

    void Relu_Act::Backward(Tensor &src, Tensor &src_grad, Tensor &dst_grad) {

        float* pSrcData = src.getCPUData();
        float* pSrcGradData = src_grad.getCPUData();
        float* pDstGradData = dst_grad.getCPUData();

        int wholeSize = src.getWholeSize();
        for (int i = 0; i < wholeSize; ++i)
        {
            pSrcGradData[i] = pDstGradData[i] *
                            ( (pSrcData[i] > 0) + negative_slope_ * (pSrcData[i] <=0) );
        }
    };

}
