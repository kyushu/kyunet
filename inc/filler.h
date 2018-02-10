#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include <random>

#include "tensor.h"
#include "filler.h"

namespace mkt {

    // class Xavier
    // {
    // public:
    //     Xavier(){};
    //     ~Xavier(){};
    //     void operator() (Tensor &x) {
    //         fprintf(stderr, "hellor xavier \n");
    //     };

    // };

    class Xavier
    {
    Distribution distri_;
    public:
        Xavier(Distribution distri):distri_{distri}{};
        ~Xavier(){};

        void operator() (Tensor &x) {

            int in = x.getNumOfData() * x.getSize2D();
            int out = x.getSize2D() * x.getDepth();
            float* xData = x.getData();
            if (distri_ == Distribution::NORM)
            {
                float sigma = 2.0f / (in + out);
                fprintf(stderr, "sigma=%f\n", sigma);
                std::default_random_engine generator;
                std::normal_distribution<float> distribution(0,sigma);
                for (int i = 0; i < x.getWholeSize(); ++i)
                {
                    xData[i] = distribution(generator);
                }
            }
            else if (distri_ == Distribution::UNIFORM) {
                float sigma = 6.0f / (in + out);

                std::default_random_engine generator;
                std::uniform_real_distribution<float> distribution(0,sigma);
                for (int i = 0; i < x.getWholeSize(); ++i)
                {
                    xData[i] = distribution(generator);
                }
            }
        }
    };

    // // He initialization aka MSRA
    // class HeInit
    // {
    //     Distribution distri_;
    // public:
    //     HeInit(Distribution: distri):distri_{distri}{};
    //     ~HeInit(){};

    //     void operator() (Tensor *x) {
    //         int in = x->getNumOfData() * x->getSize2D();
    //         int out = x->getSize2D() * x->getDepth();
    //         if (distri_ == NORM)
    //         {
    //             int sigma = 2.0f / (in);

    //             std::default_random_engine generator;
    //             std::normal_distribution<float> distribution(0,sigma);
    //             for (int i = 0; i < x->wholeSize(); ++i)
    //             {
    //                 x->pData[i] = distribution(generator);
    //             }
    //         }
    //         else if (distri_ == UNIFORM) {
    //             int sigma = 6.0f / (in);

    //             int sigma = 2.0f / (in);

    //             std::default_random_engine generator;
    //             std::normal_distribution<float> distribution(0,sigma);
    //             for (int i = 0; i < x->wholeSize(); ++i)
    //             {
    //                 x->pData[i] = distribution(generator);
    //             }
    //         }
    //     }
    // }
}



#endif