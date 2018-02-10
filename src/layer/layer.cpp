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
        mktLog(1, "--------------------- Layer Destructor\n");

        pSrc_ = nullptr;
        mktLog(1, "--------------------- Layer Destructor pSrc_\n");

        mktLog(1, "pDst_.adr: %p\n", pDst_);
        mktLog(1, "pDst_->pData.adr: %p\n", pDst_->getData());
        delete pDst_;
        mktLog(1, "--------------------- Layer Destructor pDst_\n");

        mktLog(1, "pW_.adr: %p\n", pW_);
        mktLog(1, "pW_->pData.adr: %p\n", pW_->getData());
        delete pW_;
        mktLog(1, "--------------------- Layer Destructor pW_\n");

        mktLog(1, "pB_.adr: %p\n", pB_);
        mktLog(1, "pB_->pData.adr: %p\n", pB_->getData());
        delete pB_;
        mktLog(1, "--------------------- Layer Destructor pB_\n");

    };


    // ##################################
    // Init Function
    void Layer::initOutputTensor() {
        pDst_->allocate();
    }
    void Layer::initWeightTensor() {

        pW_->allocate();
        int weight_wholeSize = pW_->getWholeSize();
        float* pWData = pW_->getData();
        switch (weightInitType_) {
            case InitializerType::ZERO:
            {
                std::fill_n(pWData, weight_wholeSize, 0);
                break;
            }
            case InitializerType::ONE:
            {
                std::fill_n(pWData, weight_wholeSize, 1);
                break;
            }
            case InitializerType::TEST:
            {
                for (int i = 0; i < weight_wholeSize; ++i)
                {
                    pWData[i] = i;
                }
                break;
            }
            case InitializerType::XAVIER_NORM:
            {

                Xavier xavier{Distribution::NORM};
                xavier(*pW_);
                break;
            }
            case InitializerType::HE_INIT_NORM:
            {
                mktLog(1, "TODO: HE_INIT\n");
                break;
            }
            default:
                mktLog(1, "Default: NO Initialize\n");
                break;
        }
    }
    void Layer::initBiasTensor() {
        pB_-> allocate();

    }

    // ##################################
    void Layer::addBias() {
        for (int i = 0; i < pDst_->getNumOfData(); ++i)
        {
            int numData = i * pDst_->getSize3D();
            axpy(pDst_->getSize3D(), 1.0, pB_->getData(), pDst_->getData()+numData);
        }
    }

    void Layer::applyActivation() {

        switch (activationType_) {
            case ActivationType::Sigmoid:
            {
                mktLog(1, "TODO: Sigmoid\n");
                break;
            }
            case ActivationType::Tanh:
            {
                mktLog(1, "TODO: Tanh\n");
                break;
            }
            case ActivationType::Relu:
            {
                mktLog(1, "TODO: Relu\n");
                break;
            }
            case ActivationType::LRelu:
            {
                mktLog(1, "TODO: LRelu\n");
                break;
            }
            case ActivationType::Selu:
            {
                mktLog(1, "TODO: Selu\n");
                break;
            }
            default:
                mktLog(1, "Default: No Activation is applied\n");
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
