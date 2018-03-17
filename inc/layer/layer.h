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

#include "operators/mat_operators.h"
#include "tensor.h"
#include "filler.hpp"
#include "activator/activator.h"
#include "activator/relu_act.h"
#include "activator/sigmoid_act.h"
#include "LayerParams.hpp"

namespace mkt {


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

        Tensor *pDst_;      // destination tensor
        Tensor* pgDst_;      // derivate data

        Tensor *pW_;        // weight tensor
        Tensor *pgW_;       // derivate data of weight

        Tensor* pB_;        // bias tensor
        Tensor* pgB_;       // derviate data of bias

        Activator* pActivator_;

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
        virtual void initialize()=0;

        void initOutputTensor();
        void initGradOutputTensor();

        void initWeightTensor();
        void initGradWeightTensor();

        void initBiasTensor();
        void initGradBiasTensor();

        // Computation Function
        virtual void Forward()=0;     // forward pass
        virtual void Backward()=0;    // back propagation

        void applyActivator();

        // Getter function
        LayerType Type();
        InitializerType Weight_Init_Type();
        InitializerType Bias_Init_Type();
        ActivationType Activation_Type();
        int BatchSize();
        int Output_Height();
        int Output_Width();
        int Output_Channel();
        int Filter_Height();
        int Filter_Width();
        int Filter_Channel();




        // Data

    };
}


#endif
