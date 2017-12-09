#include "tensor.h"
#include "inc_thirdparty.h"

// #define NDEBUG // the assert will be disabled, if NDEBUG is defined
#include    <cassert>

namespace mkt {

    /********************************
    ** Get functions
    ********************************/
    const float* Tensor::getData() {
        return data_;
    }

    const float* Tensor::getGData() {
        return gdata_;
    }

    int Tensor::getSize() {
        return size_;
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

    // add data from file
    OP_STATUS Tensor::addData(char const *filename) {

        // Safety Check
        if (data_wr_idx >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", data_wr_idx, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }
        
        // Get current write address
        float* ptr = data_ + data_wr_idx * size_;

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
            *(ptr+i) = (float)*(pImg+i);
        }

        data_wr_idx++;

        return OP_STATUS::SUCCESS;
    }

    // add data from array
    OP_STATUS Tensor::addData(const float *pImg) {

        assert(pImg);

        // Safety Check
        if (data_wr_idx >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", data_wr_idx, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        // Get current write address
        float* ptr = data_ + data_wr_idx * size_;

        for (int i = 0; i < size_; ++i)
        {
            *(ptr+i) = *(pImg+i);
        }

        data_wr_idx++;

        return OP_STATUS::SUCCESS;
    }

    // add data from array
    OP_STATUS Tensor::addData(std::vector<float> vImg) {

        // Safety Check
        if (data_wr_idx >= batchSize_)
        {
            fprintf(stderr, "cur = %d > %d\n", data_wr_idx, batchSize_);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        if (vImg.size() != size_)
        {
            return OP_STATUS::UNMATCHED_SIZE; 
        }

        // Get current write address
        float* ptr = data_ + data_wr_idx * size_;

        for (int i = 0; i < size_; ++i)
        {
            *(ptr+i) = vImg.at(i);
        }

        data_wr_idx++;

        return OP_STATUS::SUCCESS;
    }
}

