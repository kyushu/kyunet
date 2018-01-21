#ifndef _DENSELAYER_H_
#define _DENSELAYER_H_

#include "layer.h"

namespace mkt {

    class DenseLayer: public Layer
    {

    public:
        int unit;

        // Constructor with ID
        DenseLayer(
            Layer* prevLayer,
            std::string id_,
            int unit_,
            ActivationType actType_,
            InitializerType weightInitType_,
            InitializerType biasInitType_
        ): unit{unit_}, Layer(LayerType::FullConnected, actType_, weightInitType_, biasInitType_)
        {
            id = id_;

            int batchSize = prevLayer->pDst->getBatchSize();
            int h = prevLayer->pDst->getHeight();
            int w = prevLayer->pDst->getWidth();
            int c = prevLayer->pDst->getDepth();
            int size3D = prevLayer->pDst->getSize3D();

            // fprintf(stderr, "dense constructor\n");
            // fprintf(stderr, "batchSize: %d\n", batchSize);
            // fprintf(stderr, "h: %d\n", h);
            // fprintf(stderr, "w: %d\n", w);
            // fprintf(stderr, "c: %d\n", c);
            // fprintf(stderr, "size3D: %d\n", size3D);

            // pSrc_ point to pDst_ of previous layer
            pSrc = prevLayer->pDst;

            pDst = new Tensor{batchSize, 1, unit, 1};
            pW   = new Tensor{1, size3D, unit, 1};
            pB   = new Tensor{1, 1, unit, 1};

            // TODO: Activation
        };

        // Constructor without ID
        DenseLayer(
            Layer* prevLayer,
            int unit_,
            ActivationType actType_,
            InitializerType weightInitType_,
            InitializerType biasInitType_
        ): unit{unit_}, Layer(LayerType::FullConnected, actType_, weightInitType_, biasInitType_)
        {
            int batchSize = prevLayer->pDst->getBatchSize();
            int h = prevLayer->pDst->getHeight();
            int w = prevLayer->pDst->getWidth();
            int c = prevLayer->pDst->getDepth();
            int size3D = prevLayer->pDst->getSize3D();

            // pSrc_ point to pDst_ of previous layer
            pSrc = prevLayer->pDst;

            pDst = new Tensor{batchSize, 1, unit, 1};
            pW   = new Tensor{1, size3D, unit, 1};
            pB   = new Tensor{1, 1, unit, 1};

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
