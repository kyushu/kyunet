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
        int num_; // batch size
        int channel_; // channel / depth
        int height_; // height
        int width_; // width

        int size2D_;
        int size3D_;
        int wholeSize_;
    public:
        int wrIdx_;
        float *pData_;   // data


    public:
        Tensor():
            num_{0},
            channel_{0},
            height_{0},
            width_{0},
            wrIdx_{0},
            size2D_{0},
            size3D_{0},
            wholeSize_{0},
            pData_{nullptr}
        {};

        Tensor(int batchSize, int height, int width, int ch):
            num_{batchSize},
            height_{height},
            width_{width},
            channel_{ch}
        {
            size2D_ = height_ * width_;
            size3D_ = size2D_ * channel_;
            // fprintf(stderr, "tensor construct num_: %d\n", num_);
            // fprintf(stderr, "tensor construct height_: %d\n", height);
            // fprintf(stderr, "tensor construct width_: %d\n", width_);
            // fprintf(stderr, "tensor construct channel_: %d\n", channel_);
        };

        ~Tensor(){
            delete[] pData_;
        };

        // TODO: Tensor CopyConstructor
        // Copy contructor
        Tensor(const Tensor&) = delete;             // delete default copy construct
        Tensor operator=(const Tensor&) = delete;   // delete default assign
        //============================================================================


        // Initialize Function
        // void initialize(int batchSize, int h, int w, int c);
        void initialize(InitializerType init_type);

        // Add Data Function
        OP_STATUS addData(char const *filename);
        OP_STATUS addData(const float *pImg);
        OP_STATUS addData(std::vector<float> vImg);

        //
        void cleanData();

        // Getter
        const float* getData();
        int getNumOfData();
        int getDepth();
        int getWidth();
        int getHeight();
        int getSize2D();
        int getSize3D();
        int getWholeSize();


    };
}

#endif
