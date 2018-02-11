#include "activator/relu_act.h"

namespace mkt {

    // Constructor
    Relu_Act::Relu_Act() {};

    // Destructor
    Relu_Act::~Relu_Act() {};

    void Relu_Act::forward(Tensor &src, Tensor &dst) {

        // Check size
        int srcWholeSize = src.getWholeSize();
        int dstWholeSize = src.getWholeSize();
        if (!Check_EQ(srcWholeSize, dstWholeSize))
        {
            MKT_ERR_LOG("Relu:Forward, size of src(%d) and dst(%d) is not equal\n", srcWholeSize, dstWholeSize);
            return;
        }

        // Get data memory
        float* pSrcData = src.getData();
        float* pDstData = dst.getData();

        //
        for (int i = 0; i < dstWholeSize; ++i)
        {
            // negative_slop is used for leaky_relu
            pDstData[i] = std::max(pSrcData[i], 0.0f) + negative_slope_ * std::min(pSrcData[i], 0.0f);
        }

    };

    void Relu_Act::backward(Tensor &src, Tensor &src_grad, Tensor &dst_grad) {

        float* pSrcData = src.getData();
        float* pSrcGradData = src_grad.getData();
        float* pDstGradData = dst_grad.getData();

        int wholeSize = src.getWholeSize();
        for (int i = 0; i < wholeSize; ++i)
        {
            pSrcGradData[i] = pDstGradData[i] *
                            ( (pSrcData[i] > 0) + negative_slope_ * (pSrcData[i] <=0) );
        }
    };

}
