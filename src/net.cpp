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
    Net is main class to contain all layers
*/

#include "net.h"

namespace mkt {

    /****************
     *  constructor
     ****************/
    Net::Net(): pInputLayer_{nullptr} {};

    /**************
     *  Destructor
     **************/
    Net::~Net(){

        fprintf(stderr, "------------------- net destructor\n");
        fprintf(stderr, "layers.size(): %ld\n", layers_.size());
        for (int i = layers_.size()-1; i >= 0; --i)
        {
            fprintf(stderr, "delete %d\n", i);
            delete(layers_.at(i));
        }

    };

    /**************************
     *  Configuration Function
     **************************/
    Layer* Net::addInputLayer(std::string id, int batchSize, int h, int w, int c) {

        pInputLayer_ = new InputLayer{id, batchSize, h, w, c};
        layers_.push_back(pInputLayer_);

        return pInputLayer_;

    }

    Layer* Net::addDenseLayer(Layer* prevLayer, std::string id, int unit, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType) {

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

    Layer* Net::addConvLayer(
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
    Layer* Net::addConvLayer(Layer* prevLayer, std::string id, LayerParams params) {
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
    Layer* Net::addReluLayer(Layer* prevLayer, std::string id) {
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
    Layer* Net::addSigmoidLayer(Layer* prevLayer, std::string id) {
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

    Layer* Net::addPoolingLayer( Layer* prevLayer, std::string id, int kernel_Height, int kernel_width, int stride_h, int stride_w, int pad_h, int pad_w, PoolingMethodType type)
    {

        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        PoolingLayer* poolingLayer= new PoolingLayer{prevLayer, id, kernel_Height, kernel_width, stride_h, stride_w, pad_h, pad_w, type};

        layers_.push_back(poolingLayer);

        return poolingLayer;
    }

    Layer* Net::addPoolingLayer( Layer* prevLayer, std::string id, LayerParams params) {
        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }
        PoolingLayer* poolingLayer= new PoolingLayer{prevLayer, id, params};
        layers_.push_back(poolingLayer);
        return poolingLayer;
    }

    Layer* Net::addSoftmaxLayer( Layer* prevLayer, std::string id)
    {

        if (layers_.size() == 0) {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        SoftmaxLayer* softmaxLayer= new SoftmaxLayer{prevLayer, id};
        layers_.push_back(softmaxLayer);
        return softmaxLayer;
    }

    Layer* Net::addCrossEntropyLossWithSoftmaxLayer( Layer* prevLayer, std::string id)
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
     *  Initializtion Function
     **************************/
    void Net::initialize() {

        if (layers_.size() == 0) {
            return;
        }
        else {
            for (int i = 0; i < layers_.size(); ++i)
            {
                Layer* layer = layers_.at(i);
                if (i == 0 && layer->Type() == LayerType::Input)
                {
                    layer->initialize();
                } else {
                    layer->initialize();
                }
            }
        }
    }

    /**************************
     *  Forward
     **************************/
    void Net::Forward() {
        if (layers_.size() == 0) {
            return;
        } else {
            for (int i = 0; i < layers_.size(); ++i)
            {
                fprintf(stderr, "forward: %d\n", i);
                Layer* pLayer = layers_.at(i);
                if (i == 0) {
                    MKT_Assert(pLayer->Type() == LayerType::Input, "The first layer is not InputLayer");
                } else {
                    pLayer->Forward();
                }
            }
        }
    }

    /**************************
     *  Backward
     **************************/
    void Net::Backward() {
        if (layers_.size() == 0) {
            return;
        } else {
            for (int i = 0; i < layers_.size(); ++i)
            {
                Layer* pLayer = layers_.at(i);
                if (i == 0) {
                    MKT_Assert(pLayer->Type() == LayerType::Input, "The first layer is not InputLayer");
                } else {
                    pLayer->Backward();
                }
            }
        }
    }


    // Add data Function
    OP_STATUS Net::add_data_from_file_list(std::vector<std::string> fileList) {

        int inSize = fileList.size();
        int batchSize = pInputLayer_->pDst_->NumOfData();
        int tensor_h = pInputLayer_->pDst_->Height();
        int tensor_w = pInputLayer_->pDst_->Width();
        int tensor_c = pInputLayer_->pDst_->Channel();

        if (inSize != batchSize)
        {
            fprintf(stderr, "number of batchSize is not matched\n");
            return OP_STATUS::UNMATCHED_SIZE;
        }

        for (int i = 0; i < fileList.size(); ++i)
        {
            std::string file = fileList.at(i);
            int w, h, c;
            unsigned char *pImg = stbi_load(file.c_str(), &w, &h, &c, 0);

            if (pImg == nullptr)
            {
                fprintf(stderr, "no image\n");
                return OP_STATUS::FAIL;
            }
            if (tensor_h != h)
            {
                fprintf(stderr, "tensor_h:%d != input_h:%d\n", tensor_h, h);
                return OP_STATUS::UNMATCHED_SIZE;
            }
            if (tensor_w != w)
            {
                fprintf(stderr, "tensor_w: %d != input_w: %d\n", tensor_w, w);
                return OP_STATUS::UNMATCHED_SIZE;
            }
            if (tensor_c != c)
            {
                fprintf(stderr, "tensor_c: %d != input_c: %d\n", tensor_c, c);
                return OP_STATUS::UNMATCHED_SIZE;
            }

            pInputLayer_->FlattenImageToTensor(pImg, true);
        }


    }


    InputLayer* Net::getInputLayer() {
        return pInputLayer_;
    }

    // template class Net<float>;
    // template class Net<double>;
}
