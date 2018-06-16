#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <stdlib.h>

namespace mkt {

    void print_matrix(int batchSize, int channel, int height, int width, float* pData);

    void genRndPseudoData(float* pData, int num, int ch, int height, int width);
}

#endif
