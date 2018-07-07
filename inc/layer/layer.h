/*
* Morpheus
*/
/*
 * Layer is an abstract class
 *
 *                L(n-1)                               L(n)
 *         ______________________        _____________________________
 *        |            _______   |      |                  _______    |
 *        |           |      |   |      |                  |      |   |
 *    --->|           | pDst |   |----->| input( L(n-1),   | pDst |---|----->
 *        |           |______|   |  --->|        L(n-3) )  |______|   |
 *        |                      |  |   |                             |
 *        |  (ext_Src)           |  |   |                             |
 *        |______________________|  |   |_____________________________|
 *                                  |
 *                L(n-3)            |
 *         ______________________   |
 *        |            _______   |  |
 *        |           |      |   |  |
 *        |           | pDst |   |---
 *        |           |______|   |
 *        |                      |
 *        |  (ext_Src)           |
 *        |______________________|
 *
 *
 *
 *
 *
 *        ext_Src: extra input sources but not the output of previous layer
 */

#ifndef MKT_LAYER_H
#define MKT_LAYER_H

#include <iostream>
#include <string>

#include "operations/mat_operations.h"
#include "tensor.h"
#include "filler.hpp"
#include "activator/activator.h"
#include "activator/relu_act.h"
#include "activator/sigmoid_act.h"
#include "params.h"

namespace mkt {

    template<typename T>
    class Layer
    {
    protected:
        LayerType type_;
        InitializerType weightInitType_;
        InitializerType biasInitType_;
        ActivationType activationType_;


    public:
        std::string id_;
        Layer *pPrevLayer_; // point to dst_tensor of previous layer

        Tensor<T>* pDst_;      // destination tensor
        Tensor<T>* pgDst_;      // derivate data

        Tensor<T> *pW_;        // weight tensor
        Tensor<T> *pgW_;       // derivate data of weight

        Tensor<T>* pB_;        // bias tensor
        Tensor<T>* pgB_;       // derviate data of bias
        bool bUseBias;

        Activator<T>* pActivator_;

        int batchSize_;

        // tensor Dimension
        int oh_; // Dst Tensor height
        int ow_; // Dst Tensor widht
        int oc_; // Dst Tensor depth (channel)




    public:
        Layer(
            LayerType type,
            ActivationType activationType  = ActivationType::NONE,
            InitializerType weightInitType = InitializerType::NONE,
            InitializerType biasInitType   = InitializerType::NONE
        );

        virtual ~Layer();

        // TODO: copy constructor

        // Initialize Function
        virtual void initialize(NetMode mode)=0;

        void initOutputTensor();
        void initGradTensor();

        void initWeightTensor();
        void initGradWeightTensor();

        void initBiasTensor();
        void initGradBiasTensor();

        // Compute Function
        virtual void Forward()=0;     // forward pass
        virtual void Backward()=0;    // back propagation

        // archive weight and bias
        // virtual void serialize()=0;
        // virtual void deserialize()=0;

        void applyActivator();

        // Getter function
        LayerType getType();
        InitializerType getWeight_Init_Type();
        InitializerType getBias_Init_Type();
        ActivationType getActivation_Type();
        int getBatchSize();
        int getOutput_Height();
        int getOutput_Width();
        int getOutput_Channel();
        // int Filter_getHeight();
        // int Filter_getWidth();
        // int Filter_getChannel();

        Shape getWeight_Shape();



        // Data

    };
}


#endif
