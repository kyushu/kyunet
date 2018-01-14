
#include "inputLayer.h"

namespace mkt {

    void InputLayer::initialize() {
        initOutputTensor();
    }


    /*
      the "FlattenImageToTensor" will normalize each pixel of image from (0, 255)  to (-1, 1)
      and the pixel order will be reordered
      assume image is 3x4x3 = w x h x c
      the original order of pixel of image is

      0 1 2 3 4 5 6 7 8
      ------------------
      R G B R G B R G B
      R G B R G B R G B
      R G B R G B R G B
      R G B R G B R G B

      and convert it into

      0 1 2 3 4 5 6 7 8
      -----------------
      R R R R R R R R R
      R R R G G G G G G
      G G G G G G B B B
      B B B B B B B B B

    */
    void InputLayer::FlattenImageToTensor(unsigned char *pImg, bool bNormalize) {

        if (this->pDst_)
        {
            int depth  = this->pDst_->getDepth();
            int height = this->pDst_->getHeight();
            int width  = this->pDst_->getWidth();
            int sz = width*height;

            int size2D = this->pDst_->getSize2D();
            int size3D = this->pDst_->getSize3D();
            float* ptr = this->pDst_->pData_ + this->pDst_->wrIdx_ * size3D;
            fprintf(stdout, "size2D: %d\n", size2D);
            for (int i = 0; i < size3D; i+=depth)
            {
                int idx = int(i/depth);
                float maxValue = 255;
                for (int m = 0; m < depth; ++m)
                {
                    ptr[idx + size2D*m] = bNormalize ? float(pImg[i+m]) / maxValue : float(pImg[i+m]);
                }
            }

            this->pDst_->wrIdx_++;
        } else {
            // assert(this->pDst_);
        }
    }

    void InputLayer::DeFlattenImage(const float* pData, int height, int width, int channel, unsigned char *pImg) {


        int size2D = height*width;
        int size3D = size2D * channel;

        for (int i = 0; i < size3D; i+=channel)
        {
            int idx = int(i/channel);
            float maxValue = 255;
            for (int m = 0; m < channel; ++m)
            {
                int pixel = int(pData[idx + size2D*m] * maxValue);
                 pImg[i+m] = (unsigned char)pixel;
                 // fprintf(stdout, "%d\n", i+m);
            }
        }

    }

}
