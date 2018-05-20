#ifndef MKT_INPUT_LAYER_H
#define MKT_INPUT_LAYER_H

#include "layer.h"

namespace mkt {

    template<typename T>
    class InputLayer: public Layer<T>
    {
    // private:
    //     LayerType type;
    public:
        InputLayer(std::string id, int bSize, int h, int w, int c);

        ~InputLayer();

        // Method
        void initialize(NetMode mode);
        void addFlattenImageToTensor(unsigned char *pImg, int index, bool bNormalize);
        void DeFlattenImage(const T* pData, int height, int width, int channel, unsigned char *pImg);

        void Forward(){};
        void Backward(){};

    };
}


#endif
