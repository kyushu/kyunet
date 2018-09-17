#include "layer/conv_layer.h"

#include "operations/conv_operations.h"

namespace mkt {

    // Constructor
    template<typename T>
    ConvLayer<T>::ConvLayer(
        Layer<T>* prevLayer,
        std::string id,
        int fh,
        int fw,
        int fc,
        ConvParam convParam,
        ActivationType actType,
        InitializerType weightInitType,
        InitializerType biasInitType
    ):
        // kernelSize_{kernelSize},
        fh_{fh},
        fw_{fw},
        fc_{fc},
        convParam_{convParam},
        Layer<T>(LayerType::CONVOLUTION, actType, weightInitType, biasInitType)
    {

        MKT_Assert(fc_ > 0, "fc_ = 0");
        MKT_Assert(fh_ > 0, "fh_ = 0");
        MKT_Assert(fw_ > 0, "fw_ = 0");
        MKT_Assert(convParam_.stride_h_ > 0, "stride_h_ = 0");
        MKT_Assert(convParam_.stride_w_ > 0, "stride_w_ = 0");


        this->id_ = id;
        this->batchSize_ = prevLayer->pDst_->getNumOfData();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();
        // int size3D = prevLayer->pDst_->getgetSize3D();

        this->pPrevLayer_ = prevLayer;

        this->oc_ = fc_;

        // Calculate oh, ow by input and filter dimension
        // calcOutputSize(ic, ih, iw);
        InferShape();

        this->pDst_  = new Tensor<T>{this->batchSize_, this->oc_, this->oh_, this->ow_};
        this->pgDst_ = new Tensor<T>{this->batchSize_, this->oc_, this->oh_, this->ow_};

        this->pW_   = new Tensor<T>{ic, fc_, fh_, fw_};
        this->pgW_  = new Tensor<T>{ic, fc_, fh_, fw_};

        this->pB_   = new Tensor<T>{1, this->oc_, 1, 1};
        this->pgB_  = new Tensor<T>{1, this->oc_, 1, 1};


        this->pTmpCol_ = new Tensor<T>{1, 1, this->pW_->getSize2D()*ic, this->pDst_->getSize2D()};

        // Activator
        this->applyActivator();
    }

    // construct with LayerParams
    template<typename T>
    ConvLayer<T>::ConvLayer(Layer<T>* prevLayer, std::string id, LayerParams params):Layer<T>(LayerType::CONVOLUTION)
    {
        this->id_ = id;

        this->batchSize_ = prevLayer->pDst_->getNumOfData();
        int ih = prevLayer->pDst_->getHeight();
        int iw = prevLayer->pDst_->getWidth();
        int ic = prevLayer->pDst_->getChannel();

        this->pPrevLayer_ = prevLayer;

        // Parameter setting
        this->activationType_ = params.actType;
        this->weightInitType_ = params.weight_init_type;
        this->biasInitType_   = params.bias_init_type;

        fc_ = params.fc;
        fh_ = params.fh;
        fw_ = params.fw;

        convParam_.stride_h_ = params.stride_h;
        convParam_.stride_w_ = params.stride_w;

        convParam_.paddingType_ = params.padding_type;
        convParam_.pad_h_ = params.pad_h;
        convParam_.pad_w_ = params.pad_w;

        // Temporary set dilation to 1
        convParam_.dilation_h_ = params.dilation_h;
        convParam_.dilation_w_ = params.dilation_w;

        this->oc_ = fc_;

        MKT_Assert(fc_ > 0, "fc_ = 0");
        MKT_Assert(fh_ > 0, "fh_ = 0");
        MKT_Assert(fw_ > 0, "fw_ = 0");
        MKT_Assert(convParam_.stride_h_ > 0, "stride_h_ = 0");
        MKT_Assert(convParam_.stride_w_ > 0, "stride_w_ = 0");

        // Calculate oh, ow by input and filter dimension
        // calcOutputSize(ic, ih, iw);
        InferShape();

        this->pDst_  = new Tensor<T>{this->batchSize_, this->oc_, this->oh_, this->ow_};
        this->pgDst_ = new Tensor<T>{this->batchSize_, this->oc_, this->oh_, this->ow_};

        this->pW_   = new Tensor<T>{ic, fc_, fh_, fw_};
        this->pgW_  = new Tensor<T>{ic, fc_, fh_, fw_};

        this->pB_   = new Tensor<T>{1, this->oc_, 1, 1};
        this->pgB_  = new Tensor<T>{1, this->oc_, 1, 1};


        pTmpCol_ = new Tensor<T>{1, 1, this->pW_->getSize2D()*ic, this->pDst_->getSize2D()};

        // Activator
        this->applyActivator();
    }

    // Destructor
    template<typename T>
    ConvLayer<T>::~ConvLayer() {
        // fprintf(stderr, "---------------------- ConvLayer Destructor\n");
        delete pTmpCol_;
    }


    // Initialization
    template<typename T>
    void ConvLayer<T>::initialize(NetMode mode) {

        MKT_Assert(this->pDst_  != nullptr, "pDst_ is null");
        MKT_Assert(this->pgDst_ != nullptr, "pgDst_ is null");
        MKT_Assert(this->pW_    != nullptr, "pW_ is null");
        MKT_Assert(this->pgW_   != nullptr, "pgW_ is null");
        MKT_Assert(this->pB_    != nullptr, "pB_ is null");
        MKT_Assert(this->pgB_   != nullptr, "pgB_ is null");
        // MKT_Assert(this->pActivator_ != nullptr, "pActivator_ is null");
        MKT_Assert(pTmpCol_ != nullptr, "pTmpCol_ is null");

        this->initOutputTensor();
        this->initWeightTensor();
        this->initBiasTensor();

        this->initGradTensor();
        this->initGradWeightTensor();
        this->initGradBiasTensor();


        // temporary memory for im2col and col2im
        pTmpCol_->allocate();
        std::fill_n(pTmpCol_->getCPUData(), pTmpCol_->getWholeSize(), 0);


        // //TEST
        fprintf(stderr, "%s Init\n", this->id_.c_str());
        Tensor<T>* pTensor= this->pW_;
        fprintf(stderr, "pw: %p\n", pTensor);
        T* pWData = pTensor->getCPUData();
        fprintf(stderr, "pwdata: %p\n", pWData);
        // int channel   = pTensor->getChannel();
        // int height    = pTensor->getHeight();
        // int width     = pTensor->getWidth();
        // int size2D    = pTensor->getSize2D();
        // int size3D    = pTensor->getSize3D();
        // int batchSize = pTensor->getNumOfData();
        // for (int b = 0; b < batchSize; ++b)
        // {
        //     fprintf(stderr, "batch: %d\n", b);
        //     for (int c = 0; c < channel; ++c)
        //     {
        //         fprintf(stderr, "channel: %d ", c);
        //         for (int h = 0; h < height; ++h)
        //         {
        //             for (int w = 0; w < width; ++w)
        //             {
        //                 if (pWData[w + h*width + c*size2D + b*size3D] > 10.0)
        //                 {
        //                     fprintf(stderr, "pWData[%d]=%.6f\t", w + h*width + c*size2D + b*size3D, pWData[w + h*width + c*size2D + b*size3D]);
        //                 }
                        
        //             }
        //             if (height > 1) { fprintf(stderr, " "); }
        //         }
        //         if (channel > 1) { fprintf(stderr, "\n"); }
        //     }
        //     if (batchSize > 1) { fprintf(stderr, "\n"); }
        // }
        // fprintf(stderr, "\n");
        // //TEST
    }

    // Computation Function
    template<typename T>
    void ConvLayer<T>::Forward() {

        // //TEST
        // fprintf(stderr, "Forward: %s\n", this->id_.c_str());
        // Tensor<T>* pTensor= this->pW_;
        // fprintf(stderr, "pw: %p\n", pTensor);
        // T* pWData = pTensor->getCPUData();
        // fprintf(stderr, "pwdata: %p\n", pWData);
        // int channel   = pTensor->getChannel();
        // int height    = pTensor->getHeight();
        // int width     = pTensor->getWidth();
        // int size2D    = pTensor->getSize2D();
        // int size3D    = pTensor->getSize3D();
        // int batchSize = pTensor->getNumOfData();
        // for (int b = 0; b < batchSize; ++b)
        // {
        //     fprintf(stderr, "batch: %d\n", b);
        //     for (int c = 0; c < channel; ++c)
        //     {
        //         fprintf(stderr, "channel: %d ", c);
        //         for (int h = 0; h < height; ++h)
        //         {
        //             for (int w = 0; w < width; ++w)
        //             {
        //                 if (pWData[w + h*width + c*size2D + b*size3D] > 10.0)
        //                 {
        //                     fprintf(stderr, "pWData[%d]=%.6f\t", w + h*width + c*size2D + b*size3D, pWData[w + h*width + c*size2D + b*size3D]);
        //                 }
                        
        //             }
        //             if (height > 1) { fprintf(stderr, " "); }
        //         }
        //         if (channel > 1) { fprintf(stderr, "\n"); }
        //     }
        //     if (batchSize > 1) { fprintf(stderr, "\n"); }
        // }
        // fprintf(stderr, "\n");
        // //TEST

        // Reset data
        this->pDst_->resetData();
        this->pgDst_->resetData();
        this->pgW_->resetData();
        this->pgB_->resetData();



        // Apply convolutional operation
        op::convolution (
            this->batchSize_, convParam_,
            this->pPrevLayer_->pDst_, this->pDst_,
            this->pW_, this->pB_, pTmpCol_

        );

        // A = activation(Z) = the input of next layer
        if (this->activationType_ != ActivationType::NONE)
        {
            this->pActivator_->Forward(this->pDst_, this->pDst_);
        }


        // // TEST
        // T* pdst_data = this->pDst_->getCPUData();
        // for (int i = 0; i < this->pDst_->getSize3D(); ++i)
        // {
        //     fprintf(stderr, "conv_forward(%s): %d - %f\n", this->id_.c_str(), i, pdst_data[i]);
        // }
        // // TEST

    }

    template<typename T>
    void ConvLayer<T>::Backward() {

        // // TEST
        // if (this->id_ == "conv2")
        // {
        //     T* pwData = this->pW_->getCPUData();
        //     fprintf(stderr, "%p\n", pwData);
        //     for (int i = 0; i < this->pW_->getSize3D(); ++i)
        //     {
        //         fprintf(stderr, "0 conv_backward(%s): %d - %f\n", this->id_.c_str(), i, pwData[i]);
        //     }
        // }
        // // TEST



        // 1. Back from Activator first
        if (this->activationType_ != ActivationType::NONE)
        {
            this->pActivator_->Backward(this->pDst_, this->pgDst_, this->pgDst_);
        }

        // TEST
        if (this->id_ == "conv3")
        {
            fprintf(stderr, "%s  backward\n", this->id_.c_str());
                Tensor<T>* pTensor = this->pDst_;
                T* pdata      = pTensor->getCPUData();
                int size3D    = pTensor->getSize3D();
                int size2D    = pTensor->getSize2D();
                int channel   = pTensor->getChannel();
                int height    = pTensor->getHeight();
                int width     = pTensor->getWidth();
                int batchSize = pTensor->getNumOfData();

                for (int b = 0; b < batchSize; ++b)
                {
                    for (int c = 0; c < channel; ++c)
                    {
                        
                        for (int h = 0; h < height; ++h)
                        {
                            for (int w = 0; w < width; ++w)
                            {
                                if (fabs(pdata[w + h*width + c*size2D + b*size3D]) > 5.0)
                                {
                                    fprintf(stderr, "batch: %d ", b);
                                    fprintf(stderr, "channel: %d\n", c);
                                    fprintf(stderr, "pdata[%d]=%.6f\n", w + h*width + c*size2D + b*size3D, 
                                        pdata[w + h*width + c*size2D + b*size3D]);
                                }
                            }
                        }
                    }
                }
                fprintf(stderr, "\n");
        }
        
        // TEST


        // take gradient of convolutional operatoin
        op::conv_gradient (this->id_,
            this->batchSize_, convParam_, this->pW_->getShape(),
            this->pPrevLayer_->pDst_, this->pPrevLayer_->pgDst_, this->pgDst_,
            this->pW_, this->pgW_, this->pgB_, pTmpCol_
        );
    }

    template<typename T>
    void ConvLayer<T>::InferShape ()
    { 
        MKT_Assert(this->pPrevLayer_ != nullptr, "pPrevLayer_ = nullptr");

        int ih = this->pPrevLayer_->pDst_->getHeight();
        int iw = this->pPrevLayer_->pDst_->getWidth();
        

        int width_extent  = ( convParam_.dilation_w_ * (fw_ - 1) + 1 );
        int height_extent = ( convParam_.dilation_h_ * (fh_ - 1) + 1 );
        this->ow_ = static_cast<int>( static_cast<float>(iw + 2*convParam_.pad_w_ - width_extent  ) / convParam_.stride_w_ ) + 1;
        this->oh_ = static_cast<int>( static_cast<float>(ih + 2*convParam_.pad_h_ - height_extent ) / convParam_.stride_h_ ) + 1;
    }


    template<typename T>
    void ConvLayer<T>::serialize(std::fstream& fileHandler, bool bWriteInfo)
    {
        this->pW_->serialize(fileHandler);
        this->pB_->serialize(fileHandler);
    }
    
    template<typename T>
    void ConvLayer<T>::deserialize(std::fstream& fileHandler, bool bWriteInfo)
    {
        this->pW_->deserialize(fileHandler);
        this->pB_->deserialize(fileHandler);
    }

    // Getter
    template<typename T>
    int ConvLayer<T>::getFiltergetHeight()  { return fh_; }

    template<typename T>
    int ConvLayer<T>::getFiltergetWidth()   { return fw_; }

    template<typename T>
    int ConvLayer<T>::getFiltergetChannel() { return fc_; }

    template<typename T>
    Tensor<T>* ConvLayer<T>::getTmpCol()       { return pTmpCol_; }


    // Explicitly instantiate the template, and its member definitions
    template class ConvLayer<float>;

} // namespace mkt



/* #### [ Forward  Pass] ####
 *
 * ### Step 1. Im2col ###
 * For example:
 *     src     = ic(3) * ih(3) * iw(3)
 *     filter  = ic(3) * fh(2) * fw(2) * fc(2)
 *     padding = 0
 *     stride  = 1
 *     dst     = oh(2) * ow(2)
 *
 * ------------------------------------------------------------------------
 *              ch1          ch2          ch3
 * src =   | 00 01 02 | | 09 10 11 | | 18 19 20 |
 *         | 03 04 05 | | 12 13 14 | | 21 22 23 |
 *         | 06 07 08 | | 15 16 17 | | 24 25 26 |
 *
 * ------------------------------------------------------------------------
 * According to the dimension of src and filter
 * The filter(weight_matrix) is
 * Filter_0 = |w_0_000, w_0_001| |w_0_100, w_0_101| |w_0_200, w_0_201|
 *            |w_0_010, w_0_011| |w_0_110, w_0_111| |w_0_210, w_0_211|
 *
 * Filter_1 = |w_1_000, w_1_001| |w_1_100, w_1_101| |w_1_200, w_1_201|
 *            |w_1_010, w_1_011| |w_1_110, w_1_111| |w_1_210, w_1_211|
 *  w_c_ihw:
 *  c = index of filter
 *  i = index of scr channel
 *  h = index of filter height (row)
 *  w = index of filter width  (col)
 * ------------------------------------------------------------------------
 *
 * The first region of convolution operation is
 * |0 1| convert to column vector = |0|
 * |3 4|                            |1|
 *                                  |3|
 *                                  |4|
 *
 * ------------------------------------------------------------------------
 * For the computation reason, we need to convert each
 * receptive field(2d patch) of filter into col vector.
 *
 * Conver src from image to column vector
 *                      oh * ow = 2*2 = 4
 * im2col(src) =   | s00 s01 s03 s04 |
 *                 | s01 s02 s04 s05 |  src Channel 0
 *                 | s03 s04 s06 s07 |
 *                 | s04 s05 s07 s08 |____________
 *                 | s09 s10 s12 s13 |
 *                 | s10 s11 s13 s14 |  src Channel 1
 *                 | s12 s13 s15 s16 |
 *                 | s13 s14 s16 s17 |____________
 *                 | s18 s19 s21 s22 |
 *                 | s19 s20 s22 s23 |  src Channel 2
 *                 | s21 s22 s24 s25 |
 *                 | s22 s23 s25 s26 |____________
 *                    |   |   |   |
 *                    |   |   |   |__________________
 *                    |   |   |____________          |
 *                    |   |_______         |         |
 *                    |           |        |         |
 *             =   | patch_0 , patch_1, patch_2, patch_3 |
 *
 * im2col matrix = (ic*fh*fw) by (oh*ow)
 *
 *
 * ------------------------------------------------------------------------
 * ### Step 2. GEMM: kernel X im2col ###
 * All parameters follow above
 *     src     = ih(3) x iw(3) x ic(3)
 *     filter  = fc(2) x ( ic(3) * fh(2) * fw(2) )
 *     padding = 0
 *     stride  = 1
 *     dst     = oh(2) X ow(2)
 *
 * 1. Next, we will onvert filter from |w0,w1| to [w0, w1, w2, w3] (row vector)
 *                                     |w2,w3|
 *
 *     According to the dimension of src and filter
 *     The filter is
 *                     Ch 0 of src        Ch 1 of src       Ch 2 of src
 *
 *     Filter_0    |w_0_000, w_0_001| |w_0_100, w_0_101| |w_0_200, w_0_201|
 *                 |w_0_010, w_0_011| |w_0_110, w_0_111| |w_0_210, w_0_211|
 *
 *     Filter_1    |w_1_000, w_1_001| |w_1_100, w_1_101| |w_1_200, w_1_201|
 *                 |w_1_010, w_1_011| |w_1_110, w_1_111| |w_1_210, w_1_211|
 *
 *     w_c_ihw,
 *     c = index of filter
 *     i = index of scr channel
 *     h = index of filter height (row)
 *     w = index of filter width  (col)
 *
 *
 * 1-1.
 *  The filter matrix is converted to row vectors.
 *       (   Ch 0 of src   )  (   Ch 1 of src   )  (   Ch 2 of src   )
 *     | w_0_000 ... w_0_011, w_0_100 ... w_0_111, w_0_200 ... w_0_211 | = filter 0 (F0)
 *     | w_1_000 ... w_1_011, w_1_100 ... w_1_111, w_1_200 ... w_1_211 | = filter 1 (F1)
 *
 *  The sequential index of memory address is
 *       (       Ch 0 of src      )  (       Ch 1 of src      )  (       Ch 2 of src      )
 *     | w_000, w_001, w_002, w_003, w_004, w_005, w_006, w_007, w_008, w_009, w_010, w_011 |
 *     | w_012, w_013, w_014, w_015, w_016, w_017, w_018, w_019, w_020, w_021, w_022, w_023 |
 *
 *     filter matrix(Weight_matrix) = oc X (fh*fw*ic)
 *
 *
 * 2. Dst matrix = Weight_matrix X im2col(src) =
 *                          (oh * ow)
 *                     | s00 s01 s03 s04 |
 *                     | s01 s02 s04 s05 |
 *                     | s03 s04 s06 s07 |
 *                     | s04 s05 s07 s08 |
 *       (fh*fw*ic)    | s09 s10 s12 s13 |
 *       |   F0   |    | s10 s11 s13 s14 |   | d0, d1, d2, d3 |
 * (oc)  |   F1   |  X | s12 s13 s15 s16 | = | d4, d5, d6, d7 | = oc X (oh*ow)
 *                     | s13 s14 s16 s17 |
 *                     | s18 s19 s21 s22 |
 *                     | s19 s20 s22 s23 |
 *                     | s21 s22 s24 s25 |
 *                     | s22 s23 s25 s26 |
 *
 *                    ( output ch 0 )( outputch 1 )
 *     Dst matrix = | d0, d1, d2, d3 d4, d5, d6, d7 | = oc X (oh*ow)
*/


/* #### [ Backpropagation ] ####
 *
 * For backpropagation, we have to update the gradient with respect to data(feature map), weight and bias
 *
 *
 *
 * 1. w = [w0, w1, w2, w3]
 *
 * 2. dst_grad = [d0, d1, d2, d3]
 *
 *          | w0 |
 * 3. w_t = | w1 |
 *          | w2 |
 *          | w3 |
 *
 *                          | w0 |                       | w0d0, w0d1, w0d2, w0d3 |   | c0 |
 * 4. gemm(w_t, dst_grad) = | w1 |  X [d0, d1, d2, d3] = | w1d0, w1d1, w1d2, w1d3 | = | c1 |
 *                          | w2 |                       | w2d0, w2d1, w2d2, w2d3 |   | c2 |
 *                          | w3 |                       | w3d0, w3d1, w3d2, w3d3 |   | c3 |
 *
 *
 * base on reverse convolution, the src_grad_data is
 *
 *                     | w0d0      , w0d1                , w1d1      |
 * 5. src_grad_data =  | w0d2+w2d0 , w0d3+w1d2+w2d1+w3d0 , w1d3+w3d1 |
 *                     | w2d2      , w2d3+w3d2           , w3d3      |
 *
 *     c0m, c1m, c2m, c3m = convert c0, c1, c2, c3 from column vector(col) to matrix(im))
 *
 *     c0m = c0( | w0d0, w0d1, w0d2, w0d3 | ) COL_TO_IM = |w0d0 , w0d1|
 *                                                        |w0d2 , w0d3|
 *
 *     c1m = c0( | w1d0, w1d1, w1d2, w1d3 | ) COL_TO_IM = |w0d1 , w1d1|
 *                                                        |w1d2 , w1d3|
 *
 *     c2m = c0( | w2d0, w2d1, w2d2, w2d3 | ) COL_TO_IM = |w2d0 , w2d1|
 *                                                        |w2d2 , w2d3|
 *
 *     c3m = c0( | w3d0, w3d1, w3d2, w3d3 | ) COL_TO_IM = |w3d0 , w3d1|
 *                                                        |w3d2 , w3d3|
 *
 *     src_grad_data is composited by c0m, c1m, c2m, c3m
 *
 *     here display src_grad_data is composited by c0m, c1m, c2m, c3m
 *     part of c0m, c1m, c2m, c3m are overlaped.
 *
 *     Left-Top of src_grad_data      Right-Top of src_grad_data
 *         | w0d0 , w0d1|                | w0d1 , w1d1|
 *         | w0d2 , w0d3|                | w1d2 , w1d3|
 *
 *         | w2d0 , w2d1|                | w3d0 , w3d1|
 *         | w2d2 , w2d3|                | w3d2 , w3d3|
 *     Left-Bottom of src_grad_data      Right-bottom of src_grad_data
 *
 * For example
 *     src     = ih(3) * iw(3) * ic(3)
 *     filter  = ( ic(3) * fh(2) * fw(2) ) * fc(2)
 *     padding = 0
 *     stride  = 1
 *     dst     = oh(2) X ow(2)
 *
 *  Weight matrix = oc X (fh*fw*ic):
 *     | w_0_0_0, ..., w_0_0_3, w_1_0_4, ..., w_1_0_7, w_2_0_8, ..., w_2_0_11 | = filter 0 (F0)
 *     | w_0_1_0, ..., w_0_1_3, w_1_1_4, ..., w_1_1_7, w_2_1_8, ..., w_2_0_11 | = filter 1 (F1)
 *
 * Column_matrix = Transpose(weight_matrix) * dst_grad
 *
 *                        (oc)             (oh*ow)
 * (f_size2D * src_ch)  |F0, F1| x |g00, g01, g02, g03| = (f_size2D*src_ch) X (oh*ow)
 *                                 |g10, g11, g12, g13|
 *
 * Then convert Column_matrix to Image Patch
 */
