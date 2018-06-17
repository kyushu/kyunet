
#include "operations/aux_operations.h"

namespace mkt {

    void set_memory(const int N, const float alpha, float* Y) {
        if (alpha == 0) {
            memset(Y, 0, sizeof(float)*N);
        }
        else {
            for (int i = 0; i < N; ++i) {
                Y[i] = alpha;
            }
        }
    }

    void mem_copy_cpu(int size, float* pSrcData, float* pDstData) {
        for (int i = 0; i < size; ++i)
        {
            pDstData[i] = pSrcData[i];
        }
    }
}
