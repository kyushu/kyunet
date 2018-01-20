#ifndef _INPUTLAYER_H_
#define _INPUTLAYER_H_

#include "layer.h"

namespace mkt {

    class InputLayer: public Layer
    {
    // private:
    //     LayerType type;
    public:
        InputLayer(std::string id_, int bSize, int h, int w, int c): Layer(LayerType::Input) {
            id = id_;
            // this->dh_ = h;
            // this->dw_ = w;
            // this->dc_ = c;
            pDst = new Tensor{bSize, h, w, c};
            pW = new Tensor(); // empty Tensor
            pB = new Tensor(); // empty Tensor

        };

        ~InputLayer(){};

        // Method
        void initialize();
        void FlattenImageToTensor(unsigned char *pImg, bool bNormalize);
        void DeFlattenImage(const float* pData, int height, int width, int channel, unsigned char *pImg);

        void forward(){};
        void backward(){};

    };
}


#endif
