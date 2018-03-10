#include <iostream>
#include <cstdio>

#include "tensor.h"
#include "layer/layer.h"
#include "layer/input_layer.h"
#include "net.h"
#include "operators/mat_operators.h"

using namespace mkt;

void test_axpy() {
    Tensor a{2, 8, 1, 1};
    a.allocate();
    float* a_data = a.cpu_data();


    Tensor b{1, 8, 1, 1};
    b.allocate();
    float* b_data = b.cpu_data();
    for (int i = 0; i < b.WholeSize(); ++i)
    {
        b_data[i] += 3;
    }

    fprintf(stderr, "a\n");
    for (int i = 0; i < a.WholeSize(); ++i)
    {
        if (i > 0 && (i % a.Size3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%.1f ", a_data[i]);
    }

    fprintf(stderr, "\n\n");
    fprintf(stderr, "b\n");
    for (int i = 0; i < b.WholeSize(); ++i)
    {
        if (i > 0 && (i % b.Size3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%.1f ", b_data[i]);
    }

    fprintf(stderr, "\n\n");

    for (int i = 0; i < a.NumOfData(); ++i)
    {
        int size3D = a.Size3D();
        axpy(a.Size3D(), 1.0, b_data, a_data+i*size3D);
    }

    fprintf(stderr, "axpy\n");
    for (int i = 0; i < a.WholeSize(); ++i)
    {
        if (i > 0 && (i % a.Size3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%.1f ", a_data[i]);
    }

    fprintf(stderr, "\n\n");
}

int main(int argc, char const *argv[])
{

    test_axpy();



    return 0;
}
