/*
* Copyright (c) 2017 Morpheus Tsai.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*
    KyuNet is main class to contain all layers
*/

#include "net.h"
#include "solver/sgd_solver.h"

namespace mkt {

    /****************
     *  constructor
     ****************/
    template<typename T>
    KyuNet<T>::KyuNet(): pSolver_{nullptr}, pInputLayer_{nullptr} {};

    /**************
     *  Destructor
     **************/
    template<typename T>
    KyuNet<T>::~KyuNet() {

        fprintf(stderr, "------------------- net destructor\n");
        fprintf(stderr, "layers.size(): %ld\n", layers_.size());
        for (int i = layers_.size()-1; i >= 0; --i)
        {
            fprintf(stderr, "delete %d\n", i);
            delete(layers_.at(i));
        }

    };

    /**
     * Add Solver
     */
    template<typename T>
    void KyuNet<T>::addSolver(Solver<T>* pSolver) {
        pSolver_ = pSolver;
    }

    /**************************
     *  Add Layer Function
     **************************/
    template<typename T>
    Layer<T>* KyuNet<T>::addInputLayer(std::string id, int batchSize, int h, int w, int c)
    {

        pInputLayer_ = new InputLayer<T>{id, batchSize, h, w, c};
        layers_.push_back(pInputLayer_);

        return pInputLayer_;

    }

    template<typename T>
    Layer<T>* KyuNet<T>::addDenseLayer(Layer<T>* prevLayer, std::string id, int unit, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType)
    {

        if (layers_.size() == 0)
        {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        // Instantiate Dense Layer
        DenseLayer<T>* pDenseLayer = new DenseLayer<T>{prevLayer, id, unit, activationType, weightInitType, biasInitType};

        // Add layer
        layers_.push_back(pDenseLayer);

        return pDenseLayer;
    }

    template<typename T>
    Layer<T>* KyuNet<T>::addDenseLayer(Layer<T>* prevLayer, std::string id, LayerParams params)
    {
        if (layers_.size() == 0)
        {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }
        // Instantiate Dense Layer
        DenseLayer<T>* pDenseLayer = new DenseLayer<T>{prevLayer, id, params};

        // Add layer
        layers_.push_back(pDenseLayer);

        return pDenseLayer;
    }

    /* Convolution Layer */
    template<typename T>
    Layer<T>* KyuNet<T>::addConvLayer(
        Layer<T>* prevLayer,
        std::string id,
        int kernel_Height,
        int kernel_width,
        int kernel_channel,
        ConvParam convParam,
        ActivationType activationType,
        InitializerType weightInitType, InitializerType biasInitType
        )
    {

        if (layers_.size() == 0)
        {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        // Instantiate Conv Layer
        // ConvLayer<T>* pConvLayer = new ConvLayer<T>{prevLayer, id, kernel_Height, kernel_width, kernel_channel, stride_h, stride_w, pad_h, pad_w, paddingType, activationType, weightInitType, biasInitType};
        ConvLayer<T>* pConvLayer = new ConvLayer<T>{prevLayer, id, kernel_Height, kernel_width, kernel_channel, convParam, activationType, weightInitType, biasInitType};


        // Add Layer
        layers_.push_back(pConvLayer);

        return pConvLayer;
    }

    template<typename T>
    Layer<T>* KyuNet<T>::addConvLayer(Layer<T>* prevLayer, std::string id, LayerParams params) {
        if (layers_.size() == 0)
        {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        ConvLayer<T>* pConvLayer = new ConvLayer<T>{prevLayer, id, params};

        layers_.push_back(pConvLayer);
        return pConvLayer;
    }

    // Add Relu layer
    template<typename T>
    Layer<T>* KyuNet<T>::addReluLayer(Layer<T>* prevLayer, std::string id) {
        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        // Instantiate Relu Layer
        ReluLayer<T>* pReluLayer = new ReluLayer<T>(prevLayer, id);

        // Add Layer
        layers_.push_back(pReluLayer);

        return pReluLayer;
    }

    // Add Sigmoid layer
    template<typename T>
    Layer<T>* KyuNet<T>::addSigmoidLayer(Layer<T>* prevLayer, std::string id) {
        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        // Instantiate Relu Layer
        SigmoidLayer<T>* psigmoidLayer = new SigmoidLayer<T>(prevLayer, id);

        // Add Layer
        layers_.push_back(psigmoidLayer);

        return psigmoidLayer;
    }

    template<typename T>
    Layer<T>* KyuNet<T>::addPoolingLayer( Layer<T>* prevLayer, std::string id, int kernel_Height, int kernel_width, ConvParam convParam, PoolingMethodType type)
    {

        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        PoolingLayer<T>* poolingLayer= new PoolingLayer<T>{prevLayer, id, kernel_Height, kernel_width, convParam, type};

        layers_.push_back(poolingLayer);

        return poolingLayer;
    }

    template<typename T>
    Layer<T>* KyuNet<T>::addPoolingLayer( Layer<T>* prevLayer, std::string id, LayerParams params) {
        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }
        PoolingLayer<T>* poolingLayer= new PoolingLayer<T>{prevLayer, id, params};
        layers_.push_back(poolingLayer);
        return poolingLayer;
    }

    template<typename T>
    Layer<T>* KyuNet<T>::addSoftmaxLayer( Layer<T>* prevLayer, std::string id)
    {

        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        SoftmaxLayer<T>* softmaxLayer= new SoftmaxLayer<T>{prevLayer, id};
        layers_.push_back(softmaxLayer);
        return softmaxLayer;
    }

    template<typename T>
    Layer<T>* KyuNet<T>::addCrossEntropyLossWithSoftmaxLayer( Layer<T>* prevLayer, std::string id)
    {

        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        CrossEntropyLossWithSoftmaxLayer<T>* pCELossWithSoftmaxLayer= new CrossEntropyLossWithSoftmaxLayer<T>{prevLayer, id};
        layers_.push_back(pCELossWithSoftmaxLayer);
        return pCELossWithSoftmaxLayer;
    }

    template<typename T>
    Layer<T>* KyuNet<T>::addBatchNormLayer(Layer<T>* prevLayer, std::string id)
    {

        BatchNormLayer<T>* pBatchNormLayer = new BatchNormLayer<T>{prevLayer, id, InitializerType::ONE, InitializerType::ZERO};
        layers_.push_back(pBatchNormLayer);
        return pBatchNormLayer;
    }

    /**************************
     *  Compile Function
     **************************/
    template<typename T>
    void KyuNet<T>::Compile(NetMode mode) {

        if (layers_.size() == 0) {
            return;
        }

        // Initialize Layer
        for (int i = 0; i < layers_.size(); ++i)
        {
            Layer<T>* layer = layers_.at(i);
            if (i == 0 && layer->getType() == LayerType::INPUT)
            {
                layer->initialize(mode);
            } else {
                layer->initialize(mode);
            }
        }

        // Initialize Solver
        // MKT_Assert(pSolver != nullptr, "Solver is not exist");

        if (mode == NetMode::TRAINING)
        {
            MKT_Assert(pSolver_ != nullptr, "pSolver = nullptr");
            pSolver_->initialize();
        }

    }

    /**************************
     *  Forward
     **************************/
    template<typename T>
    void KyuNet<T>::Forward() {
        if (layers_.size() == 0) {
            return;
        } else {
            for (int i = 0; i < layers_.size(); ++i)
            {
                // fprintf(stderr, "forward: %d\n", i);
                Layer<T>* pLayer = layers_.at(i);
                if (i == 0) {
                    MKT_Assert(pLayer->getType() == LayerType::INPUT, "The first layer is not InputLayer");
                } else {
                    pLayer->Forward();
                }
            }
        }
    }

    /**************************
     *  Backward
     **************************/
    template<typename T>
    void KyuNet<T>::Backward() {
        if (layers_.size() == 0) {
            return;
        } else {
            // for (int i = layers_.size()-1; i > 0; --i)
            for(size_t i = layers_.size(); i-- > 0; )
            {
                Layer<T>* pLayer = layers_.at(i);
                if (i == 0) {
                    MKT_Assert(pLayer->getType() == LayerType::INPUT, "The first layer is not InputLayer");
                } else {
                    // fprintf(stderr, "backward: %ld\n", i);
                    pLayer->Backward();

                }
            }
        }
    }

    /**************************
     *  Update
     **************************/
    template<typename T>
    void KyuNet<T>::Train() {

        // every Tensor for temporary data of layer will be
        // reset in Forward function of each layer
        Forward();

        Backward();

        pSolver_->Update();

        // Clean pgDst
        // for (size_t i = 0; i < layers_.size(); ++i)
        // {
        //     Layer* pLayer = layers_.at(i);
        //     if (pLayer->pDst_)  { pLayer->pDst_->cleanData();  }
        //     if (pLayer->pgDst_) { pLayer->pgDst_->cleanData(); }
        //     if (pLayer->pgW_)   { pLayer->pgW_->cleanData();   }

        // }
    }

    // Add data Function
    template<typename T>
    OP_STATUS KyuNet<T>::add_data_from_file_list(std::vector<std::string> fileList) {


        int numImage = fileList.size();
        int batchSize = pInputLayer_->pDst_->getNumOfData();
        int tensor_h = pInputLayer_->pDst_->getHeight();
        int tensor_w = pInputLayer_->pDst_->getWidth();
        int tensor_c = pInputLayer_->pDst_->getChannel();

        if (numImage != batchSize)
        {
            fprintf(stderr, "number of batchSize is not matched\n");
            return OP_STATUS::UNMATCHED_SIZE;
        }

        // Load image from fileList
        for (int i = 0; i < fileList.size(); ++i)
        {
            std::string file = fileList.at(i);

            // Load image from file to char memory array
            int w, h, c;
            unsigned char *pImg = stbi_load(file.c_str(), &w, &h, &c, 0);

            // if (pImg == nullptr)
            // {
            //     fprintf(stderr, "no image\n");
            //     return OP_STATUS::FAIL;
            // }
            MKT_Assert(pImg != nullptr, "can't open " + file);

            // if (tensor_h != h)
            // {
            //     fprintf(stderr, "tensor_h:%d != input_h:%d\n", tensor_h, h);
            //     return OP_STATUS::UNMATCHED_SIZE;
            // }
            MKT_Assert(tensor_h == h, "tensor_h("+ std::to_string(tensor_h) + ") != " + std::to_string(h));

            // if (tensor_w != w)
            // {
            //     fprintf(stderr, "tensor_w: %d != input_w: %d\n", tensor_w, w);
            //     return OP_STATUS::UNMATCHED_SIZE;
            // }
            MKT_Assert(tensor_w == w, "tensor_w("+ std::to_string(tensor_w) + ") != " + std::to_string(w));

            // if (tensor_c != c)
            // {
            //     fprintf(stderr, "tensor_c: %d != input_c: %d\n", tensor_c, c);
            //     return OP_STATUS::UNMATCHED_SIZE;
            // }
            MKT_Assert(tensor_c == c, "tensor_w("+ std::to_string(tensor_c) + ") != " + std::to_string(c));

            pInputLayer_->addFlattenImageToTensor(pImg, i, true);
        }

        return OP_STATUS::SUCCESS;
    }

    template<typename T>
    void KyuNet<T>::addBatchLabels(std::string layer, std::vector<int> labels) {

        CrossEntropyLossWithSoftmaxLayer<T> *pCrossEntropyLayer = (CrossEntropyLossWithSoftmaxLayer<T> *)layers_.back();

        pCrossEntropyLayer->LoadLabel(labels);
    }

    /**
     * Getter function
     */
    template<typename T>
    int KyuNet<T>::getNumOfLayer() { return layers_.size(); }

    template<typename T>
    const InputLayer<T>* KyuNet<T>::getInputLayer() {
        return pInputLayer_;
    }

    template<typename T>
    const std::vector<Layer<T>*> KyuNet<T>::getLayers() {

        return layers_;
    }

    /**
     * Helper function
     */
    template<typename T>
    void KyuNet<T>::deFlattenInputImage(unsigned char *pImg, int index, int max_pixel_value) {

        int batchSize = pInputLayer_->pDst_->getNumOfData();
        if (index < batchSize)
        {
            T* pInData = pInputLayer_->pDst_->getCPUData();
            int size2D = pInputLayer_->pDst_->getSize2D();
            int size3D = pInputLayer_->pDst_->getSize3D();
            int channel = pInputLayer_->pDst_->getChannel();
            int image_offset = index * size3D;

            for (int i = 0; i < size3D; i+=channel)
            {
                int idx = int(i/channel);
                for (int c = 0; c < channel; ++c)
                {
                    int pixel = int((pInData[image_offset + idx + size2D*c]/2.0f - 0.5) * static_cast<T>(max_pixel_value) );
                    pImg[i+c] = (unsigned)pixel;
                }
            }
        } else {
            MKT_Assert(index < batchSize, "index(" + std::to_string(index) + ") > " + std::to_string(batchSize));
        }
    }

    template class KyuNet<float>;
    // template class KyuNet<double>;

} // namespace mkt
