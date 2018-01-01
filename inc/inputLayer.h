

#ifndef _INPUTLAYER_H_
#define _INPUTLAYER_H_

#include "layer.h"

namespace mkt {

    template<class DType>
    class InputLayer : public Layer<DType> 
    {
    // private:
    //     LayerType type;
    public:
        InputLayer(int h, int w, int c, int bSize): Layer<DType>(LayerType::Input), batchSize_{bSize} {
            this->dh_ = h;
            this->dw_ = w;
            this->dc_ = c;
        };
        
        ~InputLayer(){};

        // Method
        void FlattenImageToTensor(unsigned char *pImg, bool bNormalize);

        

        // Member
        int batchSize_;
    };
}


#endif