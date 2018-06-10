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

        // Must Implement virtual finctions form Layer class
        void initialize(NetMode mode);
        void Forward(){};
        void Backward(){};

        void addFlattenImageToTensor(unsigned char *pImg, int index, bool bNormalize);
        void DeFlattenImage(const T* pData, int height, int width, int channel, unsigned char *pImg);

    };
}


#endif
