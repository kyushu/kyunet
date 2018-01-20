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

    // Configuration Function
    Layer* Net::addInputLayer(std::string id, int batchSize, int h, int w, int c) {

        pInputLayer = new InputLayer{id, batchSize, h, w, c};
        layers.push_back(pInputLayer);

        return pInputLayer;

    }

    Layer* Net::addDenseLayer(std::string id, int unit, ActivationType activationType) {

        if (layers.size() == 0)
        {
            fprintf(stderr, "please add input layer first\n");
            return nullptr;
        }

        // Get Previous Layer
        Layer* prevLayer = layers.back();

        // Instantiate Dense Layer
        DenseLayer* pDenseLayer = new DenseLayer{id, activationType, prevLayer, unit};

        // Add layer
        layers.push_back(pDenseLayer);

        return pDenseLayer;
    }

    // Initializtion Function
    void Net::initialize() {

        if (layers.size() == 0)
        {
            return;

        } else {
            for (int i = 0; i < layers.size(); ++i)
            {
                Layer* layer = layers.at(i);
                if (i == 0 && layer->getType() == LayerType::Input)
                {
                    layer->initialize();
                } else {
                    layer->initialize();
                }
            }
        }
    }

    // Add data Function
    OP_STATUS Net::add_data_from_file_list(std::vector<std::string> fileList) {

        int inSize = fileList.size();
        int batchSize = pInputLayer->pDst->getBatchSize();
        int tensor_h = pInputLayer->pDst->getHeight();
        int tensor_w = pInputLayer->pDst->getWidth();
        int tensor_c = pInputLayer->pDst->getDepth();

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

            pInputLayer->FlattenImageToTensor(pImg, true);
        }


    }


    InputLayer* Net::getInputLayer() {
        return pInputLayer;
    }

    // template class Net<float>;
    // template class Net<double>;
}
