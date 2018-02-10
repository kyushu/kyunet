#include "activator/relu.h"

namespace mkt {

    void forward(Tensor &tensor) {

        int whileSize = tensor.getWholeSize();
        float* pData = tensor.getData();
        for (int i = 0; i < whileSize; ++i)
        {
            // negative_slop is used for leaky_relu
            pData[i] = std::max(pData[i], 0.0f) /*+ negative_slope * std::min(pData[i], 0)*/;
        }

    };

    void backward() {

    };



}
