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

    template<class T>
    void Tensor<T>::initTensor(int h, int w, int c, int batchSize) {
        height_ = h;
        width_ = w;
        channel_ = c;
        
        if (batchSize == 0)
        {
            fprintf(stderr, "batch size = 0\n");
            return;
        }

        batchSize_ = batchSize;
        whole_size_ = batchSize_ * size_;

        pData_ = new T[whole_size_];
        pGdata_ = new T[whole_size_];
    }

    // add data from file
    template<class T>
    OP_STATUS Tensor<T>::addData(char const *filename) {

        // Safety Check
        if (data_wr_idx_ >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", data_wr_idx_, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }
        
        // Get current write address
        float* ptr = pData_ + data_wr_idx_ * size_;

        // Load image from file
        int w, h, c;
        unsigned char *pImg = stbi_load(filename, &w, &h, &c, 0);

        if (w != width_ && h != height_ && c != channel_)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        // Conver unsigned char to float
        fprintf(stderr, "w: %d, h: %d, c: %d\n", w, h, c);
        for (int i = 0; i < size_; ++i)
        {
            *(ptr+i) = (T)*(pImg+i);
        }

        ++data_wr_idx_;

        return OP_STATUS::SUCCESS;
    }

    // add data from array
    template<class T>
    OP_STATUS Tensor<T>::addData(const T *pImg) {

        assert(pImg);

        // Safety Check
        if (data_wr_idx_ >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", data_wr_idx_, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        // Get current write address
        float* ptr = pData_ + data_wr_idx_ * size_;

        for (int i = 0; i < size_; ++i)
        {
            *(ptr+i) = *(pImg+i);
        }

        ++data_wr_idx_;

        return OP_STATUS::SUCCESS;
    }

    // add data from array
    template<class T>
    OP_STATUS Tensor<T>::addData(std::vector<T> vImg) {

        // Safety Check
        if (data_wr_idx_ >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", data_wr_idx_, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        if (vImg.size() != size_)
        {
            return OP_STATUS::UNMATCHED_SIZE; 
        }

        // Get current write address
        float* ptr = pData_ + data_wr_idx_ * size_;

        for (int i = 0; i < size_; ++i)
        {
            *(ptr+i) = vImg.at(i);
        }

        ++data_wr_idx_;

        return OP_STATUS::SUCCESS;
    }


    /********************************
    ** Get functions
    ********************************/
    template<class T>
    const T* Tensor<T>::getData() {
        return pData_;
    }

    template<class T>
    const T* Tensor<T>::getGData() {
        return pGdata_;
    }

    template<class T>
    int Tensor<T>::getFullSize() {
        return size_;
    }

    template<class T>
    int Tensor<T>::getBatchSize() {
        return batchSize_;
    }

    template<class T>
    int Tensor<T>::getWidth() {
        return width_;
    }

    template<class T>
    int Tensor<T>::getHeight() {
        return height_;
    }

    template<class T>
    int Tensor<T>::getDepth() {
        return channel_;
    }

    template class Tensor<float>;
    // template class Tensor<double>;

}

