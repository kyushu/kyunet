#ifndef _INPUT_LAYER_H_
#define _INPUT_LAYER_H_

#include "layer.h"

namespace mkt {

    class InputLayer: public Layer
    {
    // private:
    //     LayerType type;
    public:
        InputLayer(std::string id, int bSize, int h, int w, int c);

        ~InputLayer();

        // Method
        void initialize();
        void FlattenImageToTensor(unsigned char *pImg, bool bNormalize);
        void DeFlattenImage(const float* pData, int height, int width, int channel, unsigned char *pImg);

        void forward(){};
        void backward(){};

    };
}


#endif