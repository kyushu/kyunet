
#include "tensor.h"
#include "operation.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    //#############################
    // src tensor
    mkt::Tensor image(1, 4, 4, 3);
    image.initialize(mkt::InitializerType::NONE);
    int ih = image.getHeight();
    int iw = image.getWidth();
    int ic = image.getDepth();
    mktLog(2, "src data\n");
    for (int i = 0; i < image.getSize3D(); ++i) {
        image.pData_[i] = i;
    }

    for (int i = 0; i < image.getSize2D(); ++i)
    {
        if (i > 0 && i % 4 == 0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%.1f ", image.pData_[i]);
    }

    mktLog(2, "\n");

    for (int i = image.getSize2D(); i < 2* image.getSize2D(); ++i)
    {
        if (i > 0 && i % 4 == 0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%.1f ", image.pData_[i]);
    }
    mktLog(2, "\n");
    for (int i = 2* image.getSize2D(); i < 3 * image.getSize2D(); ++i)
    {
        if (i > 0 && i % 4 == 0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%.1f ", image.pData_[i]);
    }
    mktLog(2, "\n");


    //#############################
    // kernel tensor
    mkt::Tensor filter(2, 3, 3, 3);
    filter.initialize(mkt::InitializerType::NONE);
    int fh = filter.getHeight();
    int fw = filter.getWidth();
    int fc = filter.getDepth();
    for (int i = 0; i < filter.getWholeSize(); ++i)
    {
        filter.pData_[i] = 1;
    }
    mktLog(2, "filter\n");
    for (int i = 0; i < filter.getWholeSize(); ++i)
    {
        if (i > 0 && i%filter.getSize3D() == 0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%.1f ", filter.pData_[i]);
    }
    mktLog(2, "\n");


    //#############################
    // dst tensor
    int stride = 1, padding = 0;
    int dilation = 1;
    int oh = (ih - fh + 2*padding) / stride + 1;
    int ow = (iw - fw + 2*padding) / stride + 1;
    int oc = 2;

    mkt::Tensor dst(1, oh, ow, oc);
    dst.initialize(mkt::InitializerType::NONE);


    //#############################
    // column patch tensor
    mktLog(2, "filter.size2D: %d\n", filter.getSize2D());
    mktLog(2, "ic:%d\n", ic);
    mktLog(2, "dst.size2D: %d\n", dst.getSize2D());

    mkt::Tensor tempCol(1, filter.getSize2D()*ic, dst.getSize2D(), oc);
    tempCol.initialize(mkt::InitializerType::NONE);
    mktLog(2, "tempCol.getSize2D(): %d\n", tempCol.getSize2D());

    mktLog(2, "b4 im2col\n");

    //#############################
    // im2col
    mkt::im2col_cpu(image.pData_,
        image.getDepth(), image.getHeight(), image.getWidth(),
        filter.getHeight(), filter.getWidth(),
        padding, padding,
        stride, stride,
        dilation, dilation,
        tempCol.pData_);

    /****************
        DISPLAY DATA
     ****************/
    // for (int i = 0; i < tempCol.getSize2D(); ++i)
    // {
    //     if (i > 0 && i%(oh*ow) == 0)
    //     {
    //         mktLog(2, "\n");
    //     }
    //     mktLog(2, "%.1f ", tempCol.pData_[i]);
    // }
    // mktLog(2, "\n");
    // for (int i = tempCol.getSize2D(); i < 2*tempCol.getSize2D(); ++i)
    // {
    //     if (i > 0 && i%(oh*ow) == 0)
    //     {
    //         mktLog(2, "\n");
    //     }
    //     mktLog(2, "%.1f ", tempCol.pData_[i]);
    // }
    // mktLog(2, "\n");
    // for (int i = 2*tempCol.getSize2D(); i < 3*tempCol.getSize2D(); ++i)
    // {
    //     if (i > 0 && i%(oh*ow) == 0)
    //     {
    //         mktLog(2, "\n");
    //     }
    //     mktLog(2, "%.1f ", tempCol.pData_[i]);
    // }

    mktLog(2, "\n");
    mktLog(2, "M: %d\n",filter.getNumOfData());
    mktLog(2, "N: %d\n",dst.getSize2D());
    mktLog(2, "K: %d\n",filter.getSize2D()*ic);


    //#############################
    // kernel X im2col
    mkt::gemm_cpu(0, 0,                                                     /*trans_A, trans_B*/
            filter.getNumOfData(), dst.getSize2D(), filter.getSize2D()*ic,  /*M,       N,K*/
            1.0f, 1.0f,                                                     /*ALPHA,   BETA*/
            filter.pData_, filter.getSize2D()*ic,                           /*A,       lda(K)*/
            tempCol.pData_,   oh*ow,                                        /*B,       ldb(N)*/
            dst.pData_, oh*ow                                               /*C,       ldc(N)*/
    );

    for (int i = 0; i < dst.getSize3D(); ++i)
    {
        if (i > 0 && i%dst.getSize2D()==0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%f ", dst.pData_[i]);
    }

    mktLog(2, "\n");

    /****************
        DISPLAY DATA
     ****************/
    // int sum =   0  + 1  + 2  + 4  +  5 + 6  + 8  + 9  + 10 +
    //             16 + 17 + 18 + 20 + 21 + 22 + 24 + 25 + 26 +
    //             32 + 33 + 34 + 36 + 37 + 38 + 40 + 41 + 42;
    // mktLog(2, "sum:%d\n", sum);


    return 0;
}
