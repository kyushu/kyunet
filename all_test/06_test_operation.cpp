#include <iostream>
#include <cstdio>

#include "tensor.h"
#include "layer/layer.h"
#include "layer/input_layer.h"
#include "net.h"
#include "operations/mat_operations.h"

using namespace mkt;

void test_axpy() {
    Tensor<float> a{2, 1, 8, 1};
    a.allocate();
    float* a_data = a.getCPUData();


    Tensor<float> b{1, 1, 8, 1};
    b.allocate();
    float* b_data = b.getCPUData();
    for (int i = 0; i < b.getWholeSize(); ++i)
    {
        b_data[i] += 3;
    }

    fprintf(stderr, "a\n");
    for (int i = 0; i < a.getWholeSize(); ++i)
    {
        if (i > 0 && (i % a.getSize3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%.1f ", a_data[i]);
    }

    fprintf(stderr, "\n\n");
    fprintf(stderr, "b\n");
    for (int i = 0; i < b.getWholeSize(); ++i)
    {
        if (i > 0 && (i % b.getSize3D() == 0))
        {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%.1f ", b_data[i]);
    }

    fprintf(stderr, "\n\n");

    for (int i = 0; i < a.getNumOfData(); ++i)
    {
        int size3D = a.getSize3D();
        op::mat::axpy(a.getSize3D(), 1.0, b_data, a_data+i*size3D);
    }

    fprintf(stderr, "axpy\n");
    for (int i = 0; i < a.getWholeSize(); ++i)
    {
        if (i > 0 && (i % a.getSize3D() == 0))
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
