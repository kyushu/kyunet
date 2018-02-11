
#include "activator/sigmoid_act.h"

namespace mkt {
    inline float sigmoid(float x) {
        return ( 1.0f / (1.0f + exp(-x)) );
    }


    // Constructor
    Sigmoid_Act::Sigmoid_Act() {};

    // Destructor
    Sigmoid_Act::~Sigmoid_Act(){};

    void Sigmoid_Act::forward(Tensor &src, Tensor &dst) {

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

        for (int i = 0; i < dstWholeSize; ++i)
        {
            pDstData[i] = sigmoid(pSrcData[i]);
        }
    };

    void Sigmoid_Act::backward(Tensor &src, Tensor &src_grad, Tensor &dst_grad) {
    };

}


