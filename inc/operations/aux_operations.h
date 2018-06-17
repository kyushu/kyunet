#ifndef MKT_AUX_OPERATIONS_H
#define MKT_AUX_OPERATIONS_H

#include <stdio.h>
#include <string.h>

namespace mkt {

    void set_memory(const int N, const float alpha, float* Y);
    void mem_copy_cpu(int size, float* pSrcData, float* pDstData);



}

#endif
