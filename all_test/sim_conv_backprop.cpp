
#include "tensor.h"
#include "operation.h"


int main(int argc, char const *argv[])
{

    int batch_size = 1;
    int ih = 3;
    int iw = 3;
    int ic = 1;

    // set src differential tensor
    mkt::Tensor src_grad(batch_size, ih, iw, ic);
    src_grad.initialize(mkt::InitializerType::ONE);

    // Set weight tensor
    int fh = 2;
    int fw = 2;
    int oc = 1;
    mkt::Tensor weight(ic, 2, 2, oc);
    weight.initialize(mkt::InitializerType::ONE);

    // Set dst differential tensor
    mkt::Tensor dst_grad(batch_size, 2, 2, oc);
    dst_grad.initialize(mkt::InitializerType::ONE);


    int m = 1; // num of filter
    int n = weight.getSize2D() * weight.getNumOfData();
    int k = dst_grad.getSize2D();
    mkt::mktLog(1, "n: %d\n", n);
    mkt::mktLog(1, "k: %d\n", k);

    // Set temporary tensor for storing Col2Im matrix
    mkt::Tensor temp(batch_size, n, k, oc);
    temp.initialize(mkt::InitializerType::ONE);

    // backpropagation from dst.grad_data to src.grad_data
    // Step 1. dst_grad X weight
    /*

    1. w = [w0, w1, w2, w3]

    2. dst_grad = [d0, d1, d2, d3]

             | w0 |
    3. w_t = | w1 |
             | w2 |
             | w3 |

                             | w0 |                       | w0d0, w0d1, w0d2, w0d3 |   | c0 |
    4. gemm(w_t, dst_grad) = | w1 |  X [d0, d1, d2, d3] = | w1d0, w1d1, w1d2, w1d3 | = | c1 |
                             | w2 |                       | w2d0, w2d1, w2d2, w2d3 |   | c2 |
                             | w3 |                       | w3d0, w3d1, w3d2, w3d3 |   | c3 |



    base on reverse convolution, the src_grad_data is

                        | w0d0      , w0d1                , w1d1      |
    5. src_grad_data =  | w0d2+w2d0 , w0d3+w1d2+w2d1+w3d0 , w1d3+w3d1 |
                        | w2d2      , w2d3+w3d2           , w3d3      |

    c0m, c1m, c2m, c3m = convert c0, c1, c2, c3 from column vector(col) to matrix(im))
    src_grad_data is composited by c0m, c1m, c2m, c3m

    here display src_grad_data is composited by c0m, c1m, c2m, c3m
    Left-Top of src_grad_data      Right-Top of src_grad_data
        | w0d0 , w0d1|                | w0d1 , w1d1|
        | w0d2 , w0d3|                | w1d2 , w1d3|

        | w2d0 , w2d1|                | w3d0 , w3d1|
        | w2d2 , w2d3|                | w3d2 , w3d3|
    Left-Bottom of src_grad_data      Right-bottom of src_grad_data

    */
    mkt::mktLog(1, "gemm(weight, dst_grad, temp)\n");
    mkt::gemm_cpu(
        1, 0,
        n, k, m,
        1.0f, 0,
        weight.pData_, n,
        dst_grad.pData_, k,
        temp.pData_, k
    );

    // Step 2. Col2Im: perform like De-convolution and update to src_grad.pData
    mkt::mktLog(1, "col2im(temp, src_grad\n");
    int pad_h = 0;
    int pad_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    mkt::col2im_cpu(temp.pData_,
        ic, ih, iw, fh, fw,
         pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        src_grad.pData_);



    return 0;
}
