#include <iostream>
#include <cstdio>

#include "tensor.h"
#include "layer.h"
#include "inputLayer.h"
#include "net.h"
#include "operation.h"

using namespace mkt;

void test_axpy() {
    Tensor a{2, 8, 1, 1};
    a.initialize(InitializerType::TEST);

    Tensor b{1, 8, 1, 1};
    b.initialize(InitializerType::TEST);
    for (int i = 0; i < b.getWholeSize(); ++i)
    {
        b.pData_[i] += 3;
    }

    fprintf(stderr, "a\n");
    for (int i = 0; i < a.getWholeSize(); ++i)
    {
        if (i > 0 && (i % a.getSize3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%.1f ", a.pData_[i]);
    }

    fprintf(stderr, "\n\n");
    fprintf(stderr, "b\n");
    for (int i = 0; i < b.getWholeSize(); ++i)
    {
        if (i > 0 && (i % b.getSize3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%.1f ", b.pData_[i]);
    }

    fprintf(stderr, "\n\n");

    for (int i = 0; i < a.getNumOfData(); ++i)
    {
        int size3D = a.getSize3D();
        axpy(a.getSize3D(), 1.0, b.pData_, a.pData_+i*size3D);
    }

    fprintf(stderr, "axpy\n");
    for (int i = 0; i < a.getWholeSize(); ++i)
    {
        if (i > 0 && (i % a.getSize3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%.1f ", a.pData_[i]);
    }

    fprintf(stderr, "\n\n");
}

int main(int argc, char const *argv[])
{

    test_axpy();



    return 0;
}
