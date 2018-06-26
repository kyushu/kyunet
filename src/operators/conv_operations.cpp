
#include "operations/conv_operations.h"
#include <math.h>
#include "mkt_log.h"

namespace mkt {
namespace op {


    // void calcConvOutputSize (
    //     const int& inputW, const int& inputH,
    //     const int& filterW, const int& filterH,
    //     ConvParam& param, int& ootputW, int& outputH )
    // {

    //     int width_extent  = ( param.dilation_w_ * (filterW - 1) + 1 );
    //     int height_extent = ( param.dilation_h_ * (filterW - 1) + 1 );
    //     ootputW = static_cast<int>( static_cast<float>(inputW + 2*param.pad_w_ - width_extent  ) / param.stride_w_ ) + 1;
    //     outputH = static_cast<int>( static_cast<float>(inputH + 2*param.pad_h_ - height_extent ) / param.stride_h_ ) + 1;
    // }


    template<typename T>
    void convolution (
        int numOfSample,  const ConvParam& convParam,
        Tensor<T>* pSrc, Tensor<T>* pDst,
        Tensor<T>* pW, Tensor<T>* pB,
        Tensor<T>* pTmpCol)
    {

            // Src
            T* pSrcData = pSrc->getCPUData();
            int ic = pSrc->getChannel();
            int iw = pSrc->getWidth();
            int ih = pSrc->getHeight();
            int src_size3D = pSrc->getSize3D();
            int src_wholeSize = pSrc->getWholeSize();

            // Dst
            T* pDstData = pDst->getCPUData();
            int oc = pDst->getChannel();
            int oh = pDst->getHeight();
            int ow = pDst->getWidth();
            int dst_size2D = pDst->getSize2D();
            int dst_size3D = pDst->getSize3D();
            int dst_wholeSize = pDst->getWholeSize();

            // Weight
            T* pWData = pW->getCPUData();
            int fh = pW->getHeight();
            int fw = pW->getWidth();
            int fc = pW->getChannel();
            int filter_size2D = pW->getSize2D();
            int filter_wholeSize = pW->getWholeSize();

            T* pTmpColData = pTmpCol->getCPUData();

            // 1. Z = WX
            for (int i = 0; i < numOfSample; ++i)
            {

                // Step 1. im to column
                op::mat::im2col_cpu(pSrcData + i*src_size3D,
                    ic, ih, iw,
                    fh, fw,
                    convParam.pad_h_, convParam.pad_w_,
                    convParam.stride_h_, convParam.stride_w_,
                    convParam.dilation_h_, convParam.dilation_w_,
                    pTmpColData
                );

                // Step 2. Gemm column and weight
                op::mat::gemm_cpu(
                    CblasNoTrans, CblasNoTrans,         /* trans_A, trans_B */
                    fc, dst_size2D, filter_size2D*ic,   /* M,       N, K    */
                    1.0f,                               /* ALPHA            */
                    pWData, pW->getSize2D()*ic,         /* A,       lda(K)  */
                    pTmpColData,   oh*ow,               /* B,       ldb(N)  */
                    1.0f,                               /* BETA             */
                    pDstData + i*dst_size3D, oh*ow      /* C,       ldc(N)  */
                );
            }


            // 2. Z = WX + Bias
            // addBias();
            // fprintf(stderr, "addBias is not implemented: %s, %d\n", __FILE__, __LINE__);

    }

    template<typename T>
    void conv_gradient (
        int numOfSample, const ConvParam& convParam, const Shape& filterShape,
        Tensor<T>* pSrc, Tensor<T>* pgSrc, Tensor<T>* pgDst,
        Tensor<T>* pW, Tensor<T>* pgW, Tensor<T>* pgB,
        Tensor<T>* pTmpCol)
    {
        // Src
        T* pSrcData = pSrc->getCPUData();
        int src_size3D = pSrc->getSize3D();

        // Gradient Src
        T* pgSrcData = nullptr;
        int ic = 0;
        int ih = 0;
        int iw = 0;
        int gsrc_size3D = 0;
        if (pgSrc != nullptr)
        {
            pgSrcData = pgSrc->getCPUData();
            pgSrcData = pgSrc->getCPUData();
            ic = pgSrc->getChannel();
            ih = pgSrc->getHeight();
            iw = pgSrc->getWidth();
            gsrc_size3D = pgSrc->getSize3D();
        }

        // Gradient Dst
        T* pgDstData = pgDst->getCPUData();
        int gdst_size2D = pgDst->getSize2D();
        int gdst_size3D = pgDst->getSize3D();

        // Weight
        T* pWData = pW->getCPUData();
        int fh = pW->getHeight();
        int fw = pW->getWidth();
        int fc = pW->getChannel();
        int filter_size2D = pW->getSize2D();
        int num_in2filter = pW->getSize2D() * pW->getNumOfData();

        // Gradient Weight
        T* pgWData = pgW->getCPUData();

        // bias
        T* pgBias = pgB->getCPUData();

        T* pTmpColData = pTmpCol->getCPUData();



        // 2. [Update gradient with respect to Bias]
        T* pCh_grad = nullptr;
        for (int b = 0; b < numOfSample; ++b)
        {
            for (int c = 0; c < fc; ++c)
            {
                pCh_grad = pgDstData + filter_size2D*(c + b*fc);
                for (int i = 0; i < filter_size2D; ++i)
                {
                    pgBias[c] += pCh_grad[i];
                }
            }
        }

        for (int b = 0; b < numOfSample; ++b)
        {
            // 3. [Update gradient with respect to Weight]
            op::mat::im2col_cpu(pSrcData + b*src_size3D,
                ic, ih, iw,
                filterShape.height_, filterShape.width_,
                convParam.pad_h_, convParam.pad_w_,
                convParam.stride_h_, convParam.stride_w_,
                convParam.dilation_h_, convParam.dilation_w_,
                pTmpColData
            );
            op::mat::gemm_cpu(
                CblasNoTrans, CblasTrans,               /* trans_A, trans_B */
                filterShape.depth_, num_in2filter, gdst_size2D,         /* M,       N, K    */
                1.0f,                                   /* ALPHA            */
                pgDstData + b*gdst_size3D, gdst_size2D, /* A,       lda(K)  */
                pTmpColData, gdst_size2D,               /* B,       ldb(N)  */
                1.0f,                                   /* BETA             */
                pgWData, num_in2filter                  /* C,       ldc(N)  */
            );

            // 4. [Update gradient with respect to data]
            pTmpCol->resetData();
            if (pgSrc != nullptr)
            {
                op::mat::gemm_cpu(
                    CblasTrans, CblasNoTrans,       // trans_a, trans_b
                    num_in2filter, gdst_size2D, fc, // M, N, K
                    1.0f,                           // Alpha
                    pWData, gdst_size2D,            // A,       lda
                    pgDstData, fc,                  // B,       lda
                    0,                              // Beta
                    pTmpColData, fc                 // C,       lda
                );
                op::mat::col2im_cpu(
                    pTmpColData,
                    ic, ih, iw,
                    filterShape.height_, filterShape.width_,
                    convParam.pad_h_, convParam.pad_w_,
                    convParam.stride_h_, convParam.stride_w_,
                    convParam.dilation_h_, convParam.dilation_w_,
                    pgSrcData + b*gsrc_size3D
                );
            }
        }
    }



    /**
     * Explicit instantiation
     */
    template void convolution<float>(
        int, const ConvParam&,
        Tensor<float>*, Tensor<float>*,
        Tensor<float>*, Tensor<float>*,
        Tensor<float>*
    );

    template void conv_gradient<float>(
        int , const ConvParam&, const Shape&,
        Tensor<float>* , Tensor<float>* , Tensor<float>* ,
        Tensor<float>* , Tensor<float>* , Tensor<float>* ,
        Tensor<float>*);


} // namespace op
} // namespace mkt
