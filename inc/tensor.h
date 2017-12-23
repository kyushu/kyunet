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

#include <vector>
#include <cstdlib>
#include "definitions.h"

namespace mkt {
    template<class T>
    class Tensor
    {
    public:
        int batchSize_; // batch size
        int width_; // width
        int height_; // height
        int channel_; // channel / depth

        
        int full_size_;
        int size_;

    public:
        T *pData_;   // data
        T *pGdata_;  // gradient data

        int data_wr_idx_;
        int gdata_wr_idx_;

    public:
        Tensor(int width, int height, int channel):
            width_{width}, height_{height}, channel_{channel}, data_wr_idx_{0}, gdata_wr_idx_{0}, pData_{nullptr}, pGdata_{nullptr}
        {

            size_ = width_*height_*channel_;
            // full_size_ = batchSize_ * size_;

            // data_ = new T[max_size_];
            // gdata_ = new T[max_size_];
        };

        ~Tensor(){
            delete[] pData_;
            delete[] pGdata_;
        };
        
        void initTensorWithBatchSize(int batchSize);

        // 
        OP_STATUS addData(char const *filename);
        OP_STATUS addData(const T *pImg);
        OP_STATUS addData(std::vector<T> vImg);


        // Getter
        const T* getData();
        const T* getGData();
        int getFullSize();
        int getBatchSize();

        int getWidth();
        int getHeight();
        int getDepth();

        
    };
}

#endif
