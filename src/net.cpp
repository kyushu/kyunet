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

    // template<class DType>
    // void Net<DType>::FlattenImageToTensor(unsigned char *pImg, Tensor<DType> *pTensor, bool bNormalize) {

    //     int depth  = pTensor->getDepth();
    //     int height = pTensor->getHeight();
    //     int width  = pTensor->getWidth();
    //     int sz = width*height;

    //     int wr_idx = pTensor->data_wr_idx_;
    //     int full_size = pTensor->getFullSize();
    //     float* ptr = pTensor->pData_ + pTensor->data_wr_idx_ * full_size;

    //     for (int i = 0; i < full_size; i+=depth)
    //     {
    //         int idx = int(i/depth);
    //         DType maxValue = 255;
    //         ptr[idx]                = bNormalize ? DType(pImg[i])   / maxValue : DType(pImg[i]);
    //         ptr[sz*(depth-2) + idx] = bNormalize ? DType(pImg[i+1]) / maxValue : DType(pImg[i+1]);
    //         ptr[sz*(depth-1) + idx] = bNormalize ? DType(pImg[i+2]) / maxValue : DType(pImg[i+2]);
    //     }
    // }

    template<class DType>
    void Net<DType>::addLayer(Layer<DType> *newLayer) {

        layers_.push_back(newLayer);
    }


    template<class DType>
    void Net<DType>::compile() {

        if (layers_.size() == 0)
        {
            return;
        } else if (layers_.size() == 1)
        {
            if (layers_.at(0)->getType() == LayerType::Input)
            {
                fprintf(stderr, "the first layer is Input layer\n");
                pInputLayer = static_cast<InputLayer<DType>*>(layers_.at(0));
                // fprintf(stderr, "pInputLayer.batchSize_: %d\n", pInputLayer->batchSize_);
                int h = pInputLayer->dh_;
                int w = pInputLayer->dw_;
                int c = pInputLayer->dc_;
                pInputLayer->pDstTensor_->initTensor(h, w, c, pInputLayer->batchSize_);

            }
        } else {
            fprintf(stderr, "TODO: compile layer more than 1\n");
        }

    }


    template class Net<float>;
    // template class Net<double>;
}
