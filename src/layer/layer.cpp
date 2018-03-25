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

#include "layer/layer.h"

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
        id_{""},
        batchSize_{0},
        oh_{0}, ow_{0}, oc_{0},
        pPrevLayer_{nullptr},
        pDst_{nullptr},
        pgDst_{nullptr},
        pW_{nullptr},
        pgW_{nullptr},
        pB_{nullptr},
        pgB_{nullptr},
        pActivator_{nullptr}
    {};

    Layer::~Layer() {
        mktLog(1, "--------------------- Layer Destructor\n");

        pPrevLayer_ = nullptr;
        mktLog(1, "--------------------- Layer Destructor pPrevLayer_\n");


        mktLog(1, "pDst_.adr: %p\n", pDst_);
        if (pDst_) {mktLog(1, "pDst_->pData.adr: %p\n", pDst_->getCPUData());}
        delete pDst_;
        mktLog(1, "--------------------- Layer Destructor pDst_\n");

        mktLog(1, "pgDst_.adr: %p\n", pgDst_);
        if (pgDst_) {mktLog(1, "pgDst_->pData.adr: %p\n", pgDst_->getCPUData());}
        delete pgDst_;
        mktLog(1, "--------------------- Layer Destructor pgDst_\n");


        mktLog(1, "pW_.adr: %p\n", pW_);
        if (pW_) {mktLog(1, "pW_->pData.adr: %p\n", pW_->getCPUData());}
        delete pW_;
        mktLog(1, "--------------------- Layer Destructor pW_\n");


        mktLog(1, "pB_.adr: %p\n", pB_);
        if (pB_) {mktLog(1, "pB_->pData.adr: %p\n", pB_->getCPUData());}
        delete pB_;
        mktLog(1, "--------------------- Layer Destructor pB_\n");

        if (pActivator_)
        {
            delete pActivator_;
        }
        mktLog(1, "--------------------- Layer Destructor pActivator_\n");
    };


    /***********************************
     * Init Function
     ***********************************/
    // Tensor for forward pass
    void Layer::initOutputTensor() {
        pDst_->allocate();
    }
    void Layer::initWeightTensor() {

        pW_->allocate();
        int weight_wholeSize = pW_->getWholeSize();
        float* pWData = pW_->getCPUData();
        switch (weightInitType_) {
            case InitializerType::ZERO:
            {
                std::fill_n(pWData, weight_wholeSize, 0.0f);
                break;
            }
            case InitializerType::ONE:
            {
                std::fill_n(pWData, weight_wholeSize, 1.0f);
                break;
            }
            case InitializerType::TEST:
            {
                for (int i = 0; i < weight_wholeSize; ++i)
                {
                    pWData[i] = float(i+1);
                }
                break;
            }
            case InitializerType::XAVIER_NORM:
            {

                Xavier xavier{Distribution::NORM};
                xavier(*pW_);
                break;
            }
            case InitializerType::XAVIER_UNIFORM:
            {

                Xavier xavier{Distribution::UNIFORM};
                xavier(*pW_);
                break;
            }
            case InitializerType::HE_INIT_NORM:
            {
                HeInit he{Distribution::NORM};
                he(*pW_);
                break;
            }
            default:
                HeInit he{Distribution::UNIFORM};
                he(*pW_);
                break;
        }
    }

    // Tensor for backpropagation (gradient)
    void Layer::initGradTensor() {
        pgDst_->allocate();
        std::fill_n(pgDst_->getCPUData(), pgDst_->getWholeSize(), 0.0f);
    }
    void Layer::initGradWeightTensor() {
        pgW_->allocate();
        std::fill_n(pgW_->getCPUData(), pgW_->getWholeSize(), 0.0f);
    }

    void Layer::initBiasTensor() {
        pB_-> allocate();
        std::fill_n(pB_->getCPUData(), pB_->getWholeSize(), 0.0f);
    }

    void Layer::initGradBiasTensor() {
        pgB_->allocate();
        std::fill_n(pgB_->getCPUData(), pgB_->getWholeSize(), 0.0f);
    }

    // ##################################
    // void Layer::addBias() {
    //     for (int i = 0; i < pDst_->getNumOfData(); ++i)
    //     {
    //         int numData = i * pDst_->getSize3D();
    //         axpy(pDst_->getSize3D(), 1.0, pB_->getCPUData(), pDst_->getCPUData()+numData);
    //     }
    // }

    void Layer::applyActivator() {
        switch (activationType_) {
            case ActivationType::RELU:
            fprintf(stderr, "relu\n");
                pActivator_ = new Relu_Act{};
                break;
            case ActivationType::SIGMOID:
                fprintf(stderr, "sigmoid\n");
                pActivator_ = new Sigmoid_Act{};
                break;
            default:
                fprintf(stderr, "no activator\n");
                break;
        }
    }

    //##################################
    // Getter Function
    LayerType Layer::Type() {
        return type_;
    }
    InitializerType Layer::Weight_Init_Type(){
        return weightInitType_;
    }
    InitializerType Layer::Bias_Init_Type(){
        return biasInitType_;
    }
    ActivationType Layer::Activation_Type() {
        return activationType_;
    }

    int Layer::BatchSize() {
        return batchSize_;
    }

    int Layer::Output_getHeight() {
        return oh_;
    }
    int Layer::Output_getWidth() {
        return ow_;
    }
    int Layer::Output_getChannel() {
        return oc_;
    }


    // template class Layer<float>;
}
