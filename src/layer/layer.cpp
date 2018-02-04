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

#include "layer.h"

namespace mkt {

    Layer::Layer(
        LayerType type,
        ActivationType activationType,
        InitializerType weightInitType,
        InitializerType biasInitType
    ):
        type_{type},
        activationType_{activationType},
        weightInitType_{weightInitType},
        biasInitType_{biasInitType},
        batchSize_{0},
        oh_{0}, ow_{0}, oc_{0},
        fh_{0}, fw_{0}, fc_{0},
        pSrc_{nullptr},
        pDst_{nullptr},
        pW_{nullptr},
        pB_{nullptr}
    {};

    Layer::~Layer() {
        fprintf(stderr, "--------------------- Layer Destructor\n");

        pSrc_ = nullptr;
        fprintf(stderr, "--------------------- Layer Destructor pSrc_\n");

        fprintf(stderr, "pDst_.adr: %p\n", pDst_);
        fprintf(stderr, "pDst_->pData.adr: %p\n", pDst_->pData_);
        delete pDst_;
        fprintf(stderr, "--------------------- Layer Destructor pDst_\n");

        fprintf(stderr, "pW_.adr: %p\n", pW_);
        fprintf(stderr, "pW_->pData.adr: %p\n", pW_->pData_);
        delete pW_;
        fprintf(stderr, "--------------------- Layer Destructor pW_\n");

        fprintf(stderr, "pB_.adr: %p\n", pB_);
        fprintf(stderr, "pB_->pData.adr: %p\n", pB_->pData_);
        delete pB_;
        fprintf(stderr, "--------------------- Layer Destructor pB_\n");

    };


    // ##################################
    // Init Function
    void Layer::initOutputTensor() {
        pDst_->initialize(InitializerType::NONE);
    }
    void Layer::initWeightTensor(InitializerType initType) {
        pW_->initialize(initType);
    }
    void Layer::initBiasTensor(InitializerType initType) {
        pB_-> initialize(initType);
    }

    // ##################################
    void Layer::addBias() {
        for (int i = 0; i < pDst_->getNumOfData(); ++i)
        {
            int numData = i * pDst_->getSize3D();
            axpy(pDst_->getSize3D(), 1.0, pB_->pData_, pDst_->pData_+numData);
        }
    }

    void Layer::applyActivation() {

        switch (activationType_) {
            case ActivationType::Sigmoid:
            {
                fprintf(stderr, "TODO: Sigmoid\n");
                break;
            }
            case ActivationType::Tanh:
            {
                fprintf(stderr, "TODO: Tanh\n");
                break;
            }
            case ActivationType::Relu:
            {
                fprintf(stderr, "TODO: Relu\n");
                break;
            }
            case ActivationType::LRelu:
            {
                fprintf(stderr, "TODO: LRelu\n");
                break;
            }
            case ActivationType::Selu:
            {
                fprintf(stderr, "TODO: Selu\n");
                break;
            }
            default:
                fprintf(stderr, "Default: No Activation is applied\n");
                break;
        }

    }

    //##################################
    // Getter Function
    LayerType Layer::getType() {
        return type_;
    }
    InitializerType Layer::getWeightInitType(){
        return weightInitType_;
    }
    InitializerType Layer::getBiasInitType(){
        return biasInitType_;
    }
    ActivationType Layer::getActivationType() {
        return activationType_;
    }

    int Layer::getBatchSize() {
        return batchSize_;
    }

    int Layer::getOutputHeight() {
        return oh_;
    }
    int Layer::getOutputWidth() {
        return ow_;
    }
    int Layer::getOutputChannel() {
        return oc_;
    }

    int Layer::getFilterHeight() {
        return fh_;
    }

    int Layer::getFilterWidth() {
        return fw_;
    }

    int Layer::getFilterChannel() {
        return fc_;
    }
    // template class Layer<float>;
}
