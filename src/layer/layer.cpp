/*
 * Layer is an abstract class which only defines essential methods
*/

#include "layer/layer.h"

namespace mkt {

    template<typename T>
    Layer<T>::Layer(
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

    template<typename T>
    Layer<T>::~Layer() {
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
    template<typename T>
    void Layer<T>::initOutputTensor() {
        pDst_->allocate();
        std::fill_n(pDst_->getCPUData(), pDst_->getWholeSize(), 0);
    }

    template<typename T>
    void Layer<T>::initWeightTensor() {

        pW_->allocate();
        int weight_wholeSize = pW_->getWholeSize();
        T* pWData = pW_->getCPUData();
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
                    pWData[i] = static_cast<T>( (i+1) );
                }
                break;
            }
            case InitializerType::XAVIER_NORM:
            {

                Xavier<T> xavier{Distribution::NORM};
                xavier(*pW_);
                break;
            }
            case InitializerType::XAVIER_UNIFORM:
            {

                Xavier<T> xavier{Distribution::UNIFORM};
                xavier(*pW_);
                break;
            }
            case InitializerType::HE_INIT_NORM:
            {
                HeInit<T> he{Distribution::NORM};
                he(*pW_);
                break;
            }
            default:
                HeInit<T> he{Distribution::UNIFORM};
                he(*pW_);
                break;
        }
    }

    // Tensor for backpropagation (gradient)
    template<typename T>
    void Layer<T>::initGradTensor() {
        pgDst_->allocate();
        std::fill_n(pgDst_->getCPUData(), pgDst_->getWholeSize(), 0);
    }

    template<typename T>
    void Layer<T>::initGradWeightTensor() {
        pgW_->allocate();
        std::fill_n(pgW_->getCPUData(), pgW_->getWholeSize(), 0);
    }

    template<typename T>
    void Layer<T>::initBiasTensor() {
        pB_-> allocate();
        std::fill_n(pB_->getCPUData(), pB_->getWholeSize(), 0);
    }

    template<typename T>
    void Layer<T>::initGradBiasTensor() {
        pgB_->allocate();
        std::fill_n(pgB_->getCPUData(), pgB_->getWholeSize(), 0);
    }

    // ##################################
    // void Layer::addBias() {
    //     for (int i = 0; i < pDst_->getNumOfData(); ++i)
    //     {
    //         int numData = i * pDst_->getSize3D();
    //         axpy(pDst_->getSize3D(), 1.0, pB_->getCPUData(), pDst_->getCPUData()+numData);
    //     }
    // }

    template<typename T>
    void Layer<T>::applyActivator() {
        switch (activationType_) {
            case ActivationType::RELU:
            fprintf(stderr, "file: %s func: %s relu\n", __FILE__, __func__);
                pActivator_ = new Relu_Act<T>{};
                break;
            case ActivationType::SIGMOID:
                fprintf(stderr, "file: %s, func: %s: sigmoid\n", __FILE__, __func__);
                pActivator_ = new Sigmoid_Act<T>{};
                break;
            default:
                fprintf(stderr, "no activator\n");
                break;
        }
    }

    //##################################
    // Getter Function
    template<typename T>
    LayerType       Layer<T>::getType()             { return type_;           }

    template<typename T>
    InitializerType Layer<T>::getWeight_Init_Type() { return weightInitType_; }

    template<typename T>
    InitializerType Layer<T>::getBias_Init_Type()   { return biasInitType_;   }

    template<typename T>
    ActivationType  Layer<T>::getActivation_Type()  { return activationType_; }

    template<typename T>
    int             Layer<T>::getBatchSize()        { return batchSize_;      }

    template<typename T>
    int             Layer<T>::getOutput_Height()    { return oh_;             }

    template<typename T>
    int             Layer<T>::getOutput_Width()     { return ow_;             }

    template<typename T>
    int             Layer<T>::getOutput_Channel()   { return oc_;             }

    template<typename T>
    Shape           Layer<T>::getWeight_Shape()     { return pW_->getShape(); }


    // Explicitly instantiate the template, and its member definitions
    template class Layer<float>;

} // namespace mkt
