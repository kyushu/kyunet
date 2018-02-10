
#include "inputLayer.h"

namespace mkt {

    static int MAX_PIXEL_VALUE = 255;

    // Constructor
    InputLayer::InputLayer(std::string id, int bSize, int h, int w, int c): Layer(LayerType::Input) {
        id_ = id;
        // this->dh_ = h;
        // this->dw_ = w;
        // this->dc_ = c;
        pDst_ = new Tensor{bSize, h, w, c};
        pW_ = new Tensor(); // empty Tensor
        pB_ = new Tensor(); // empty Tensor

    };

    // Destructor
    InputLayer::~InputLayer() {
        fprintf(stderr, "--------------------- InputLayer Destructor\n");
    };


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

        if (pDst_)
        {
            int depth  = pDst_->getDepth();
            int height = pDst_->getHeight();
            int width  = pDst_->getWidth();
            int sz = width*height;

            int size2D = pDst_->getSize2D();
            int size3D = pDst_->getSize3D();
            float* ptr = pDst_->getData() + pDst_->wrIdx_ * size3D;
            // fprintf(stdout, "size2D: %d\n", size2D);
            for (int i = 0; i < size3D; i+=depth)
            {
                int idx = int(i/depth);
                for (int m = 0; m < depth; ++m)
                {
                    ptr[idx + size2D*m] = bNormalize ? (float(pImg[i+m]) / MAX_PIXEL_VALUE - 0.5f) * 2.0f : float(pImg[i+m]);
                }
            }

            pDst_->wrIdx_++;
        } else {
            // assert(this->pDst);
        }
    }

    // For Debug
    void InputLayer::DeFlattenImage(const float* pData, int height, int width, int channel, unsigned char *pImg) {


        int size2D = height*width;
        int size3D = size2D * channel;

        for (int i = 0; i < size3D; i+=channel)
        {
            int idx = int(i/channel);
            // float maxValue = 255;
            for (int m = 0; m < channel; ++m)
            {
                // int pixel = int(pData[idx + size2D*m] * MAX_PIXEL_VALUE);
                int pixel = int((pData[idx + size2D*m]/2.0f -0.5) * MAX_PIXEL_VALUE);

                 pImg[i+m] = (unsigned char)pixel;
                 // fprintf(stdout, "%d\n", i+m);
            }
        }

    }

}
