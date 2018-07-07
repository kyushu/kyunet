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
#include <fstream>

#include "definitions.h"
#include "params.h"

namespace mkt {
    template<typename T>
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
        T *pData_;   // data
    public:
        int wrCount_;



    public:
        Tensor();
        Tensor(int num, int ch, int height, int width);
        Tensor(Shape shape);
        ~Tensor();

        // TODO: Tensor CopyConstructor
        // Copy contructor
        Tensor(const Tensor&) = delete;             // delete default copy construct
        Tensor operator=(const Tensor&) = delete;   // delete default assign
        //============================================================================


        // Initialize Function
        void allocate();
        void Reshape(int num, int ch, int height, int width, bool reAllocate=true);

        // Add Data Function
        // Add full data to Tensor
        OP_STATUS addData(const T *pImg, int size);
        OP_STATUS addData(const std::vector<T> vImg);

        OP_STATUS addOneSample(char const *filename);
        OP_STATUS addOneSample(const T *pImg, int size);
        OP_STATUS addOneSample(const std::vector<T> vImg);

        // archive weight and bias
        void serialize(std::fstream& file, bool bWriteInfo);
        void deserialize(std::fstream& file, bool bReadInfo);

        void resetData();

        // Getter
        Shape getShape();
        T* getCPUData();
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
