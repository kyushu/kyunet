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

    // Constructor
    Tensor::Tensor():
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

    // Constructor
    Tensor::Tensor(int num, int height, int width, int ch):
        num_{num},
        height_{height},
        width_{width},
        channel_{ch}
    {
        size2D_ = height_ * width_;
        size3D_ = size2D_ * channel_;
        wholeSize_ = num_ * size3D_;
        // fprintf(stderr, "tensor construct num_: %d\n", num_);
        // fprintf(stderr, "tensor construct height_: %d\n", height);
        // fprintf(stderr, "tensor construct width_: %d\n", width_);
        // fprintf(stderr, "tensor construct channel_: %d\n", channel_);
    };

    Tensor::Tensor(Shape shape) {
        num_ = shape[0];
        height_ = shape[1];
        width_ = shape[2];
        channel_ = shape[3];

         size2D_ = height_ * width_;
        size3D_ = size2D_ * channel_;
        wholeSize_ = num_ * size3D_;
    }

    // Destructor
    Tensor::~Tensor(){
        delete[] pData_;
    };

    // Init Tensor: allocate memory space of pData

    /*************
     * Initialize
     *************/
    void Tensor::allocate() {

        // fprintf(stderr, "init_type: %d\n", init_type);
        // size2D_ = height_ * width_;
        // size3D_ = size2D_ * channel_;
        // wholeSize_ = num_ * size3D_;

        if (wholeSize_ == 0)
        {
            fprintf(stderr, "wholeSize == 0\n");
            return;
        }

        pData_ = new float[wholeSize_];

    }

    void Tensor::Reshape(int num, int height, int width, int ch) {
        num_ = num;
        height_ = height;
        width_ = width;
        channel_ = ch;

        size2D_ = height_ * width_;
        size3D_ = size2D_ * channel_;
        wholeSize_ = num_ * size3D_;

        if (pData_)
        {
            delete[] pData_;
        }

        allocate();
    }

    /*********************
     * add data from file
     *********************/
    OP_STATUS Tensor::addData(char const *filename) {

        // Safety Check
        if (wrIdx_ >= num_)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx_, num_);
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
        if (wrIdx_ >= num_)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx_, num_);
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
        if (wrIdx_ >= num_)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx_, num_);
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


    void Tensor::resetData() {
        // std::memset(pData_, 0, wholeSize_ * sizeof(float));
        std::fill_n(pData_, wholeSize_, 0.0f);
    }

    /********************************
    ** Getter
    ********************************/
    float* Tensor::getCPUData()   { return pData_; }
    int    Tensor::getNumOfData() { return num_; }
    int    Tensor::getWidth()     { return width_; }
    int    Tensor::getHeight()    { return height_; }
    int    Tensor::getChannel()   { return channel_; }
    int    Tensor::getSize2D()    { return size2D_; }
    int    Tensor::getSize3D()    { return size3D_; }
    int    Tensor::getWholeSize() { return wholeSize_; }

    Shape Tensor::getShape() {
        Shape shape{num_, height_, width_, channel_};
        return shape;
    }
}

