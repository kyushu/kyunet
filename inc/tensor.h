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


#ifndef MKT_TENSOR_H
#define MKT_TENSOR_H

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "definitions.h"

namespace mkt {
    class Tensor
    {
    private:
        int num_; // batch size
        int channel_; // channel / depth
        int height_; // height
        int width_; // width

        int size2D_;
        int size3D_;
        int wholeSize_;
        float *pData_;   // data
    public:
        int wrIdx_;



    public:
        Tensor();
        Tensor(int num, int height, int width, int ch);
        Tensor(Shape shape);
        ~Tensor();

        // TODO: Tensor CopyConstructor
        // Copy contructor
        Tensor(const Tensor&) = delete;             // delete default copy construct
        Tensor operator=(const Tensor&) = delete;   // delete default assign
        //============================================================================


        // Initialize Function
        // void initialize(int batchSize, int h, int w, int c);
        void allocate();
        void Reshape(int num, int height, int width, int ch);

        // Add Data Function
        OP_STATUS addData(char const *filename);
        OP_STATUS addData(const float *pImg);
        OP_STATUS addData(std::vector<float> vImg);

        //
        void resetData();

        // Getter
        Shape getShape();
        float* getCPUData();
        int getNumOfData();
        int getChannel();
        int getWidth();
        int getHeight();
        int getSize2D();
        int getSize3D();
        int getWholeSize();


    };
}

#endif
