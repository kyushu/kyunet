
#include "tensor.h"
#include "operators/mat_operators.h"

#include "definitions.h"

using namespace mkt;

int main(int argc, char const *argv[])
{

    //#############################
    // src tensor
    mkt::Tensor image(1, 4, 4, 3);
    image.allocate();
    float* pImgData = image.cpu_data();
    int ih = image.Height();
    int iw = image.Width();
    int ic = image.Channel();
    mktLog(2, "src data\n");
    for (int i = 0; i < image.Size3D(); ++i) {
        pImgData[i] = i;
    }

    for (int i = 0; i < image.Size2D(); ++i)
    {
        if (i > 0 && i % 4 == 0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%.1f ", pImgData[i]);
    }

    mktLog(2, "\n");

    for (int i = image.Size2D(); i < 2* image.Size2D(); ++i)
    {
        if (i > 0 && i % 4 == 0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%.1f ", pImgData[i]);
    }
    mktLog(2, "\n");
    for (int i = 2* image.Size2D(); i < 3 * image.Size2D(); ++i)
    {
        if (i > 0 && i % 4 == 0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%.1f ", pImgData[i]);
    }
    mktLog(2, "\n");


    //#############################
    // kernel tensor
    mkt::Tensor filter(2, 3, 3, 3);
    filter.allocate();
    float* pFilterData = filter.cpu_data();
    int fh = filter.Height();
    int fw = filter.Width();
    int fc = filter.Channel();
    for (int i = 0; i < filter.WholeSize(); ++i)
    {
        pFilterData[i] = 1;
    }
    mktLog(2, "filter\n");
    for (int i = 0; i < filter.WholeSize(); ++i)
    {
        if (i > 0 && i%filter.Size3D() == 0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%.1f ", pFilterData[i]);
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
    dst.allocate();
    float* pDstData = dst.cpu_data();


    //#############################
    // column patch tensor
    mktLog(2, "filter.size2D: %d\n", filter.Size2D());
    mktLog(2, "ic:%d\n", ic);
    mktLog(2, "dst.size2D: %d\n", dst.Size2D());

    mkt::Tensor tempCol(1, filter.Size2D()*ic, dst.Size2D(), oc);
    tempCol.allocate();
    float* pTmpColData = tempCol.cpu_data();
    mktLog(2, "tempCol.getSize2D(): %d\n", tempCol.Size2D());

    mktLog(2, "b4 im2col\n");

    //#############################
    // im2col
    mkt::im2col_cpu(pImgData,
        image.Channel(), image.Height(), image.Width(),
        filter.Height(), filter.Width(),
        padding, padding,
        stride, stride,
        dilation, dilation,
        pTmpColData);

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
    mktLog(2, "M: %d\n",filter.NumOfData());
    mktLog(2, "N: %d\n",dst.Size2D());
    mktLog(2, "K: %d\n",filter.Size2D()*ic);


    //#############################
    // kernel X im2col
    mkt::gemm_cpu(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,                                                     /*trans_A, trans_B*/
            filter.NumOfData(), dst.Size2D(), filter.Size2D()*ic,  /*M,       N,K*/
            1.0f,                                                           /*ALPHA*/
            pFilterData, filter.Size2D()*ic,                             /*A,       lda(K)*/
            pTmpColData,   oh*ow,                                           /*B,       ldb(N)*/
            1.0f,                                                           /*BETA*/
            pDstData, oh*ow                                                 /*C,       ldc(N)*/
    );

    for (int i = 0; i < dst.Size3D(); ++i)
    {
        if (i > 0 && i%dst.Size2D()==0)
        {
            mktLog(2, "\n");
        }
        mktLog(2, "%f ", pDstData[i]);
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
