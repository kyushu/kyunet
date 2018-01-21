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

    /*************
     * Initialize
     *************/
    void Tensor::initialize(InitializerType init_type) {

        fprintf(stderr, "init_type: %d\n", init_type);

        wholeSize = batchSize * size3D;

        if (wholeSize == 0)
        {
            fprintf(stderr, "wholeSize == 0\n");
            return;
        }

        pData = new float[wholeSize];

        //MT:TEST
        if (init_type == InitializerType::TEST)
        {
            // fprintf(stderr, "size2D_: %d\n", size2D);
            // fprintf(stderr, "size3D_: %d\n", size3D);
            // std::fill_n(pData, wholeSize, 10);
            for (int i = 0; i < wholeSize; ++i)
            {
                pData[i] = i;
            }
        }
        else if (init_type == InitializerType::RANDOM)
        {

        }
        else if(init_type == InitializerType::XAVIER) {

        }
        else if(init_type == InitializerType::HE_INIT) {

        }

    }

    /*********************
     * add data from file
     *********************/
    OP_STATUS Tensor::addData(char const *filename) {

        // Safety Check
        if (wrIdx >= batchSize)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx, batchSize);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        // Get current write address
        float* ptr = pData + wrIdx * size3D;

        // Load image from file
        int w, h, c;
        unsigned char *pImg = stbi_load(filename, &w, &h, &c, 0);

        if (w != width && h != height && c != channel)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        // Conver unsigned char to float
        fprintf(stderr, "w: %d, h: %d, c: %d\n", w, h, c);
        for (int i = 0; i < size3D; ++i)
        {
            *(ptr+i) = (float)*(pImg+i);
        }

        ++wrIdx;

        return OP_STATUS::SUCCESS;
    }

    // add data from array
    OP_STATUS Tensor::addData(const float *pImg) {

        assert(pImg);

        // Safety Check
        if (wrIdx >= batchSize)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx, batchSize);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        // Get current write address
        float* ptr = pData + wrIdx * size3D;

        for (int i = 0; i < size3D; ++i)
        {
            *(ptr+i) = *(pImg+i);
        }

        ++wrIdx;

        return OP_STATUS::SUCCESS;
    }

    // add data from vector
    OP_STATUS Tensor::addData(std::vector<float> vImg) {

        // Safety Check
        if (wrIdx >= batchSize)
        {
            fprintf(stderr, "cur = %d > %d\n", wrIdx, batchSize);
            return OP_STATUS::OVER_MAX_SIZE;
        }

        if (vImg.size() != size3D)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        // Get current write address
        float* ptr = pData + wrIdx * size3D;

        for (int i = 0; i < size3D; ++i)
        {
            *(ptr+i) = vImg.at(i);
        }

        ++wrIdx;

        return OP_STATUS::SUCCESS;
    }


    void Tensor::cleanData() {
        std::memset(pData, 0, wholeSize * sizeof(float));
    }

    /********************************
    ** Get functions
    ********************************/
    const float* Tensor::getData() {
        return pData;
    }

    int Tensor::getBatchSize() {
        return batchSize;
    }

    int Tensor::getWidth() {
        return width;
    }

    int Tensor::getHeight() {
        return height;
    }

    int Tensor::getDepth() {
        return channel;
    }

    int Tensor::getSize2D() {
        return size2D;
    }

    int Tensor::getSize3D() {
        return size3D;
    }

    int Tensor::getWholeSize() {
        return wholeSize;
    }
}

