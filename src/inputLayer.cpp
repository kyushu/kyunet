
#include "inputLayer.h"

namespace mkt {

    template<class DType>
    void InputLayer<DType>::FlattenImageToTensor(unsigned char *pImg, bool bNormalize) {

        if (this->pDstTensor_)
        {
            int depth  = this->pDstTensor_->getDepth();
            int height = this->pDstTensor_->getHeight();
            int width  = this->pDstTensor_->getWidth();
            int sz = width*height;

            int wr_idx = this->pDstTensor_->data_wr_idx_;
            int full_size = this->pDstTensor_->getFullSize();
            float* ptr = this->pDstTensor_->pData_ + this->pDstTensor_->data_wr_idx_ * full_size;

            for (int i = 0; i < full_size; i+=depth)
            {
                int idx = int(i/depth);
                DType maxValue = 255;
                ptr[idx]                = bNormalize ? DType(pImg[i])   / maxValue : DType(pImg[i]);
                ptr[sz*(depth-2) + idx] = bNormalize ? DType(pImg[i+1]) / maxValue : DType(pImg[i+1]);
                ptr[sz*(depth-1) + idx] = bNormalize ? DType(pImg[i+2]) / maxValue : DType(pImg[i+2]);
            }
        } else {
            assert(this->pDstTensor_);
        }

        
    }

}