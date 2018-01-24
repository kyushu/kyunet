#ifndef _DENSELAYER_H_
#define _DENSELAYER_H_

#include "layer.h"

namespace mkt {

    class DenseLayer: public Layer
    {

    public:
        int unit_;

        // Constructor with ID
        DenseLayer(
            Layer* prevLayer,
            std::string id,
            int unit,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        ): unit_{unit}, Layer(LayerType::FullConnected, actType, weightInitType, biasInitType)
        {
            id = id;

            int batchSize = prevLayer->pDst_->getNumOfData();
            int h = prevLayer->pDst_->getHeight();
            int w = prevLayer->pDst_->getWidth();
            int c = prevLayer->pDst_->getDepth();
            int size3D = prevLayer->pDst_->getSize3D();

            // fprintf(stderr, "dense constructor\n");
            // fprintf(stderr, "batchSize: %d\n", batchSize);
            // fprintf(stderr, "h: %d\n", h);
            // fprintf(stderr, "w: %d\n", w);
            // fprintf(stderr, "c: %d\n", c);
            // fprintf(stderr, "size3D: %d\n", size3D);

            // pSrc_ point to pDst_ of previous layer
            pSrc_ = prevLayer->pDst_;

            pDst_ = new Tensor{batchSize, 1, unit, 1};
            pW_   = new Tensor{size3D, unit, 1, 1};
            pB_   = new Tensor{1, 1, unit, 1};

            // TODO: Activation
        };

        // Constructor without ID
        DenseLayer(
            Layer* prevLayer,
            int unit,
            ActivationType actType,
            InitializerType weightInitType,
            InitializerType biasInitType
        ): unit_{unit}, Layer(LayerType::FullConnected, actType, weightInitType, biasInitType)
        {
            int batchSize = prevLayer->pDst_->getNumOfData();
            int h = prevLayer->pDst_->getHeight();
            int w = prevLayer->pDst_->getWidth();
            int c = prevLayer->pDst_->getDepth();
            int size3D = prevLayer->pDst_->getSize3D();

            // pSrc_ point to pDst_ of previous layer
            pSrc_ = prevLayer->pDst_;

            pDst_ = new Tensor{batchSize, 1, unit, 1};
            pW_   = new Tensor{1, size3D, unit, 1};
            pB_   = new Tensor{1, 1, unit, 1};

            // TODO: Activation
        };

        ~DenseLayer(){};

        void initialize();

        // Computation Function
        void forward();
        void backward();
    };

}

#endif
