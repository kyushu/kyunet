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

#ifndef _NET_H_
#define _NET_H_

// #include <cstdio>

#include "stb_image.h"

#include "layer/layer.h"
#include "layer/input_layer.h"
#include "layer/dense_layer.h"
#include "layer/conv_layer.h"


namespace mkt {

    class Net
    {
    private:
        std::vector<Layer* > layers;
        InputLayer* pInputLayer;

    public:
        //==================================
        Net();
        ~Net();
        //==================================

        // Configuration Function
        Layer* addInputLayer(std::string id, int batchSize, int h, int w, int c);
        Layer* addDenseLayer(Layer* prevLayer, std::string id, int unit, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType);
        Layer* addConvLayer(Layer* prevLayer, std::string id, int nfilter, int kernelSize, int stride, int padding, PaddingType paddingType, ActivationType activationType, InitializerType weightInitType, InitializerType biasInitType);

        // Initialize Function
        void initialize();

        // Add Data Function
        OP_STATUS add_data_from_file_list(std::vector<std::string> fileList);

        // Getter
        InputLayer* getInputLayer();

    };

}

#endif
