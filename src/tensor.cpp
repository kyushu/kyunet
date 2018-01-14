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

#include "tensor.h"
#include "inc_thirdparty.h"



namespace mkt {

    // Init Tensor: allocate memory space of pData
    // void Tensor::initialize(int batchSize, int h, int w, int c) {
    void Tensor::initialize() {

        // if (batchSize == 0)
        // {
        //     fprintf(stderr, "batch size = 0\n");
        //     return;
        // }

        // height_ = h;
        // width_ = w;
        // channel_ = c;

        size2D_ = height_ * width_;
        size3D_ = size2D_ * channel_;
        // fprintf(stderr, "channel_: %d\n", channel_);
        // fprintf(stderr, "size2D_: %d\n", size2D_);
        // fprintf(stderr, "init tensor size3D: %d\n", size3D_);
        wholeSize_ = batchSize_ * size3D_;

        if (wholeSize_ == 0)
        {
            fprintf(stderr, "wholeSize == 0\n");
            return;
        }


        pData_ = new float[wholeSize_];
    }

    // add data from file
    OP_STATUS Tensor::addData(char const *filename) {

        // Safety Check
        if (wrIdx_ >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx_, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        // Get current write address
        float* ptr = pData_ + wrIdx_ * size3D_;

        // Load image from file
        int w, h, c;
        unsigned char *pImg = stbi_load(filename, &w, &h, &c, 0);

        if (w != width_ && h != height_ && c != channel_)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        // Conver unsigned char to float
        fprintf(stderr, "w: %d, h: %d, c: %d\n", w, h, c);
        for (int i = 0; i < size3D_; ++i)
        {
            *(ptr+i) = (float)*(pImg+i);
        }

        ++wrIdx_;

        return OP_STATUS::SUCCESS;
    }

    // add data from array
    OP_STATUS Tensor::addData(const float *pImg) {

        assert(pImg);

        // Safety Check
        if (wrIdx_ >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx_, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        // Get current write address
        float* ptr = pData_ + wrIdx_ * size3D_;

        for (int i = 0; i < size3D_; ++i)
        {
            *(ptr+i) = *(pImg+i);
        }

        ++wrIdx_;

        return OP_STATUS::SUCCESS;
    }

    // add data from vector
    OP_STATUS Tensor::addData(std::vector<float> vImg) {

        // Safety Check
        if (wrIdx_ >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx_, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        if (vImg.size() != size3D_)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        // Get current write address
        float* ptr = pData_ + wrIdx_ * size3D_;

        for (int i = 0; i < size3D_; ++i)
        {
            *(ptr+i) = vImg.at(i);
        }

        ++wrIdx_;

        return OP_STATUS::SUCCESS;
    }


    /********************************
    ** Get functions
    ********************************/
    const float* Tensor::getData() {
        return pData_;
    }

    int Tensor::getBatchSize() {
        return batchSize_;
    }

    int Tensor::getWidth() {
        return width_;
    }

    int Tensor::getHeight() {
        return height_;
    }

    int Tensor::getDepth() {
        return channel_;
    }

    int Tensor::getSize2D() {
        return size2D_;
    }

    int Tensor::getSize3D() {
        return size3D_;
    }

    int Tensor::getWholdSize() {
        return wholeSize_;
    }
}

