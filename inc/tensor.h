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


#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "definitions.h"

namespace mkt {
    class Tensor
    {
    public:
        int batchSize; // batch size
        int channel; // channel / depth
        int height; // height
        int width; // width

        int size2D;
        int size3D;
        int wholeSize;
    public:
        int wrIdx;
        float *pData;   // data


    public:
        Tensor():
            batchSize{0},
            channel{0},
            height{0},
            width{0},
            wrIdx{0},
            size2D{0},
            size3D{0},
            wholeSize{0},
            pData{nullptr}
        {};

        Tensor(int batchSize_, int height_, int width_, int ch_):
            batchSize{batchSize_},
            height{height_},
            width{width_},
            channel{ch_}
        {
            size2D = height * width;
            size3D = size2D * channel;
            // fprintf(stderr, "tensor construct batchSize_: %d\n", batchSize_);
            // fprintf(stderr, "tensor construct height_: %d\n", height);
            // fprintf(stderr, "tensor construct width_: %d\n", width_);
            // fprintf(stderr, "tensor construct channel_: %d\n", channel_);
        };

        ~Tensor(){
            delete[] pData;
        };

        // TODO: Tensor CopyConstructor
        // Copy contructor
        Tensor(const Tensor&) = delete;             // delete default copy construct
        Tensor operator=(const Tensor&) = delete;   // delete default assign
        //============================================================================


        // Initialize Function
        // void initialize(int batchSize, int h, int w, int c);
        void initialize(Initializer_Type init_type=Initializer_Type::NONE);

        // Add Data Function
        OP_STATUS addData(char const *filename);
        OP_STATUS addData(const float *pImg);
        OP_STATUS addData(std::vector<float> vImg);

        //
        void cleanData();

        // Getter
        const float* getData();
        int getBatchSize();
        int getDepth();
        int getWidth();
        int getHeight();
        int getSize2D();
        int getSize3D();
        int getWholeSize();


    };
}

#endif
