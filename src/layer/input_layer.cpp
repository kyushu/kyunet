
#include "layer/input_layer.h"

namespace mkt {

    static int MAX_PIXEL_VALUE = 255;

    // Constructor
    template<typename T>
    InputLayer<T>::InputLayer(std::string id, int bSize, int h, int w, int c): Layer<T>(LayerType::INPUT) {
        this->id_ = id;
        // this->dh_ = h;
        // this->dw_ = w;
        // this->dc_ = c;
        this->pDst_ = new Tensor<T>{ bSize, c, h, w };

    };

    // Destructor
    template<typename T>
    InputLayer<T>::~InputLayer() {
        fprintf(stderr, "--------------------- InputLayer Destructor\n");
    };

    template<typename T>
    void InputLayer<T>::initialize(NetMode mode) {
        this->initOutputTensor();
    }

    /*
      the "FlattenImageToTensor" will normalize the intensity value of each pixel from (0, 255)  to (-1, 1)
      and separate RGB channel.

      assume image is 3x4x3 = w x h x c
      the original order of pixel of image is

      0 1 2 3 4 5 6 7 8
      ------------------
      R G B R G B R G B
      R G B R G B R G B
      R G B R G B R G B
      R G B R G B R G B

      and convert it into c x w x h

      0 1 2 3 4 5 6 7 8
      -----------------
      R R R R R R R R R
      R R R G G G G G G
      G G G G G G B B B
      B B B B B B B B B

    */
    template<typename T>
    void InputLayer<T>::addFlattenImageToTensor(unsigned char *pImg, int index, bool bNormalize) {

        if (this->pDst_)
        {
            int depth  = this->pDst_->getChannel();
            int height = this->pDst_->getHeight();
            int width  = this->pDst_->getWidth();
            int sz = width*height;

            int size2D = this->pDst_->getSize2D();
            int size3D = this->pDst_->getSize3D();
            T* ptr = this->pDst_->getCPUData() + index * size3D;
            for (int i = 0; i < size3D; i+=depth)
            {
                int idx = int(i/depth);
                for (int d = 0; d < depth; ++d)
                {
                    ptr[idx + size2D*d] = bNormalize ? (static_cast<T>( pImg[i+d] ) / MAX_PIXEL_VALUE - 0.5f) * 2.0f :
                                                        static_cast<T>( pImg[i+d] );
                }
            }
        }
    }

    // For Debug
    template<typename T>
    void InputLayer<T>::DeFlattenImage(const T* pData, int height, int width, int channel, unsigned char *pImg) {

        int size2D = height*width;
        int size3D = size2D * channel;

        for (int i = 0; i < size3D; i+=channel)
        {
            int idx = int(i/channel);
            for (int m = 0; m < channel; ++m)
            {
                // int pixel = int(pData[idx + size2D*m] * MAX_PIXEL_VALUE);
                int pixel = int((pData[idx + size2D*m]/2.0f - 0.5) * MAX_PIXEL_VALUE);

                 pImg[i+m] = (unsigned char)pixel;
                 // fprintf(stdout, "%d\n", i+m);
            }
        }

    }

    // Explicitly instantiate the template, and its member definitions
    template class InputLayer<float>;

} // namespace mkt

/**
 * The format of image data after FlattenImage process is
 * For instance
 * batch size     = 4
 * RGB image size = 3x2x3 = row x col x depth(channel)
 *
 * Tensor = batch_size x depth x row x col

 * [p_0_0_00, p_0_0_01, p_0_0_10, p_0_0_11, p_0_0_20, p_0_0_21, p_1_1_00 ~ p_1_1_21, p_0_2_00 ~ p_0_2_21]
 * [p_1_0_00, p_1_0_01, p_1_0_10, p_1_0_11, p_1_0_20, p_1_0_21, p_1_1_00 ~ p_1_1_21, p_1_2_00 ~ p_1_2_21]
 *
 *
 * |                 Ch 0 of image 0                          |  Ch 1 of image 0   |  Ch 2 of image 0   |
 * |                 Ch 0 of image 1                          |  Ch 1 of image 1   |  Ch 2 of image 1   |
 *
 * p_b_d_hw
 * b = batch size
 * d = depth
 * h = height = row
 * w = width  = col
 */
