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

    // ##################################
    // Init Function
    void Layer::initOutputTensor() {
        pDst->initialize(InitializerType::NONE);
    }
    void Layer::initWeightTensor(InitializerType initType) {
        pW->initialize(initType);
    }
    void Layer::initBiasTensor(InitializerType initType) {
        pB-> initialize(initType);
    }

    // ##################################
    void Layer::addBias() {
        for (int i = 0; i < pDst->getBatchSize(); ++i)
        {
            int batch = i * pDst->getSize3D();
            axpy(pDst->getSize3D(), 1.0, pB->pData, pDst->pData+batch);
        }
    }


    //##################################
    // Getter Function
    LayerType Layer::getType() {
        return type;
    }
    InitializerType Layer::getInitType(){
        return initType;
    }
    ActivationType Layer::getActivationType() {
        return activationType;
    }

    // template class Layer<float>;
}
