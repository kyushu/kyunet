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
    KyuNet::KyuNet(): pSolver_{nullptr}, pInputLayer_{nullptr} {};

    /**************
     *  Destructor
     **************/
    KyuNet::~KyuNet() {

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
    void KyuNet::addSolver(Solver* pSolver) {
        pSolver_ = pSolver;
    }

    /**************************
     *  Add Layer Function
     **************************/
    Layer* KyuNet::addInputLayer(std::string id, int batchSize, int h, int w, int c)
    {

        pInputLayer_ = new InputLayer{id, batchSize, h, w, c};
        layers_.push_back(pInputLayer_);

        return pInputLayer_;

    }

    Layer* KyuNet::addDenseLayer(Layer* prevLayer, std::string id, int unit, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType)
    {

        if (layers_.size() == 0)
        {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        // Instantiate Dense Layer
        DenseLayer* pDenseLayer = new DenseLayer{prevLayer, id, unit, activationType, weightInitType, biasInitType};

        // Add layer
        layers_.push_back(pDenseLayer);

        return pDenseLayer;
    }

    Layer* KyuNet::addDenseLayer(Layer* prevLayer, std::string id, LayerParams params)
    {
        if (layers_.size() == 0)
        {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }
        // Instantiate Dense Layer
        DenseLayer* pDenseLayer = new DenseLayer{prevLayer, id, params};

        // Add layer
        layers_.push_back(pDenseLayer);

        return pDenseLayer;
    }

    /* Convolution Layer */
    Layer* KyuNet::addConvLayer(
        Layer* prevLayer,
        std::string id,
        int kernel_Height,
        int kernel_width,
        int kernel_channel,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        PaddingType paddingType,
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
        ConvLayer* pConvLayer = new ConvLayer{prevLayer, id, kernel_Height, kernel_width, kernel_channel, stride_h, stride_w, pad_h, pad_w, paddingType, activationType, weightInitType, biasInitType};

        // Add Layer
        layers_.push_back(pConvLayer);

        return pConvLayer;
    }
    Layer* KyuNet::addConvLayer(Layer* prevLayer, std::string id, LayerParams params) {
        if (layers_.size() == 0)
        {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        ConvLayer* pConvLayer = new ConvLayer{prevLayer, id, params};

        layers_.push_back(pConvLayer);
        return pConvLayer;
    }

    // Add Relu layer
    Layer* KyuNet::addReluLayer(Layer* prevLayer, std::string id) {
        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        // Instantiate Relu Layer
        ReluLayer* pReluLayer = new ReluLayer(prevLayer, id);

        // Add Layer
        layers_.push_back(pReluLayer);

        return pReluLayer;
    }

    // Add Sigmoid layer
    Layer* KyuNet::addSigmoidLayer(Layer* prevLayer, std::string id) {
        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        // Instantiate Relu Layer
        SigmoidLayer* psigmoidLayer = new SigmoidLayer(prevLayer, id);

        // Add Layer
        layers_.push_back(psigmoidLayer);

        return psigmoidLayer;
    }

    Layer* KyuNet::addPoolingLayer( Layer* prevLayer, std::string id, int kernel_Height, int kernel_width, int stride_h, int stride_w, int pad_h, int pad_w, PoolingMethodType type)
    {

        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        PoolingLayer* poolingLayer= new PoolingLayer{prevLayer, id, kernel_Height, kernel_width, stride_h, stride_w, pad_h, pad_w, type};

        layers_.push_back(poolingLayer);

        return poolingLayer;
    }

    Layer* KyuNet::addPoolingLayer( Layer* prevLayer, std::string id, LayerParams params) {
        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }
        PoolingLayer* poolingLayer= new PoolingLayer{prevLayer, id, params};
        layers_.push_back(poolingLayer);
        return poolingLayer;
    }

    Layer* KyuNet::addSoftmaxLayer( Layer* prevLayer, std::string id)
    {

        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        SoftmaxLayer* softmaxLayer= new SoftmaxLayer{prevLayer, id};
        layers_.push_back(softmaxLayer);
        return softmaxLayer;
    }

    Layer* KyuNet::addCrossEntropyLossWithSoftmaxLayer( Layer* prevLayer, std::string id)
    {

        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        CrossEntropyLossWithSoftmaxLayer* pCELossWithSoftmaxLayer= new CrossEntropyLossWithSoftmaxLayer{prevLayer, id};
        layers_.push_back(pCELossWithSoftmaxLayer);
        return pCELossWithSoftmaxLayer;
    }

    /**************************
     *  Compile Function
     **************************/
    void KyuNet::Compile(NetMode mode) {

        if (layers_.size() == 0) {
            return;
        }

        // Initialize Layer
        for (int i = 0; i < layers_.size(); ++i)
        {
            Layer* layer = layers_.at(i);
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
    void KyuNet::Forward() {
        if (layers_.size() == 0) {
            return;
        } else {
            for (int i = 0; i < layers_.size(); ++i)
            {
                // fprintf(stderr, "forward: %d\n", i);
                Layer* pLayer = layers_.at(i);
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
    void KyuNet::Backward() {
        if (layers_.size() == 0) {
            return;
        } else {
            // for (int i = layers_.size()-1; i > 0; --i)
            for(size_t i = layers_.size(); i-- > 0; )
            {
                Layer* pLayer = layers_.at(i);
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
    void KyuNet::Train() {
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
    OP_STATUS KyuNet::add_data_from_file_list(std::vector<std::string> fileList) {


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

    void KyuNet::addBatchLabels(std::string layer, std::vector<int> labels) {

        CrossEntropyLossWithSoftmaxLayer *pCrossEntropyLayer = (CrossEntropyLossWithSoftmaxLayer *)layers_.back();

        pCrossEntropyLayer->LoadLabel(labels);
    }

    /**
     * Getter function
     */
    int KyuNet::getNumOfLayer() { return layers_.size(); }

    const InputLayer* KyuNet::getInputLayer() {
        return pInputLayer_;
    }

    const std::vector<Layer*> KyuNet::getLayers() {

        return layers_;
    }

    /**
     * Helper function
     */
    void KyuNet::deFlattenInputImage(unsigned char *pImg, int index, float max_pixel_value) {

        int batchSize = pInputLayer_->pDst_->getNumOfData();
        if (index < batchSize)
        {
            float* pInData = pInputLayer_->pDst_->getCPUData();
            int size2D = pInputLayer_->pDst_->getSize2D();
            int size3D = pInputLayer_->pDst_->getSize3D();
            int channel = pInputLayer_->pDst_->getChannel();
            int image_offset = index * size3D;

            for (int i = 0; i < size3D; i+=channel)
            {
                int idx = int(i/channel);
                for (int c = 0; c < channel; ++c)
                {
                    int pixel = int((pInData[image_offset + idx + size2D*c]/2.0f - 0.5) * max_pixel_value);
                    pImg[i+c] = (unsigned)pixel;
                }
            }
        } else {
            MKT_Assert(index < batchSize, "index(" + std::to_string(index) + ") > " + std::to_string(batchSize));
        }
    }

    // template class KyuNet<float>;
    // template class KyuNet<double>;
}
