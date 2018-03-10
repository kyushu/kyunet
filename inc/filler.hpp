#ifndef MKT_FILLER_H
#define MKT_FILLER_H

#include <random>

#include "tensor.h"

namespace mkt {

    class Xavier
    {
    Distribution distri_;
    public:
        Xavier(Distribution distri):distri_{distri}{};
        ~Xavier(){};

        void operator() (Tensor &x) {

            int in = x.NumOfData() * x.Size2D();
            int out = x.Size2D() * x.Channel();
            float* xData = x.cpu_data();
            if (distri_ == Distribution::NORM)
            {
                float sigma = 2.0f / (in + out);
                fprintf(stderr, "sigma=%f\n", sigma);
                std::default_random_engine generator;
                std::normal_distribution<float> distribution(0,sigma);
                for (int i = 0; i < x.WholeSize(); ++i)
                {
                    xData[i] = distribution(generator);
                }
            }
            else if (distri_ == Distribution::UNIFORM) {
                float sigma = 6.0f / (in + out);

                std::default_random_engine generator;
                std::uniform_real_distribution<float> distribution(0,sigma);
                for (int i = 0; i < x.WholeSize(); ++i)
                {
                    xData[i] = distribution(generator);
                }
            }
        }
    };

    // He initialization aka MSRA
    class HeInit
    {
        Distribution distri_;
    public:
        HeInit(Distribution distri):distri_{distri}{};
        ~HeInit(){};

        void operator() (Tensor &x) {
            int in = x.NumOfData() * x.Size2D();
            int out = x.Size2D() * x.Channel();
            float* xData = x.cpu_data();
            if (distri_ == Distribution::NORM)
            {
                float sigma = 2.0f / (in);

                std::default_random_engine generator;
                std::normal_distribution<float> distribution(0,sigma);
                for (int i = 0; i < x.WholeSize(); ++i)
                {
                    xData[i] = distribution(generator);
                }
            }
            else if (distri_ == Distribution::UNIFORM) {
                float sigma = 6.0f / (in);

                std::default_random_engine generator;
                std::uniform_real_distribution<float> distribution(0,sigma);
                for (int i = 0; i < x.WholeSize(); ++i)
                {
                    xData[i] = distribution(generator);
                }
            }
        }
    };
}



#endif
