#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <vector>
#include <cstdlib>
#include "definitions.h"

namespace mkt {

    class Tensor
    {
    private:
        int batchSize_; // batch size
        int width_; // width
        int height_; // height
        int channel_; // channel / depth

        
        int max_size_;
        int size_;

    public:
        float *data_;   // data
        float *gdata_;  // gradient data

        int data_wr_idx;
        int gdata_wr_idx;

    public:
        Tensor(int batch, int width, int height, int channel) {

            data_wr_idx = 0;    // the unit is width_*height_*channel_
            gdata_wr_idx = 0;   // the unit is width_*height_*channel_

            batchSize_ = batch;
            width_ = width;
            height_ = height;
            channel_ = channel;

            size_ = width_*height_*channel_;
            max_size_ = batchSize_ * size_;

            data_ = new float[max_size_];
            gdata_ = new float[max_size_];
        };

        ~Tensor(){
            delete[] data_;
            delete[] gdata_;
        };
        
        // Getter
        const float* getData();
        const float* getGData();
        int getSize();
        int getBatchSize();
        int getWidth();
        int getHeight();
        int getDepth();

        // 
        OP_STATUS addData(char const *filename);
        OP_STATUS addData(const float *pImg);
        OP_STATUS addData(std::vector<float> vImg);
    };
}

#endif
