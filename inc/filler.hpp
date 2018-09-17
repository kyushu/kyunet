#ifndef MKT_FILLER_HPP
#define MKT_FILLER_HPP

#include <random>

#include "tensor.h"

namespace mkt {

    template<typename T>
    class Xavier
    {
        Distribution distri_;
    public:
        Xavier(Distribution distri):distri_{distri}{};
        ~Xavier(){};

        void operator() (int numInData, int numOutData, Tensor<T> &x) {

            int in = x.getNumOfData() * x.getSize2D();
            int out = x.getSize2D() * x.getChannel();
            float* xData = x.getCPUData();
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

    // He initialization aka MSRA
    template<typename T>
    class HeInit
    {
        Distribution distri_;
    public:
        HeInit(Distribution distri):distri_{distri}{};
        ~HeInit(){};

        void operator() (int numInData, Tensor<T> &x) {

            int in = x.getNumOfData() * x.getSize2D();
            float* xData = x.getCPUData();
            if (distri_ == Distribution::NORM)
            {
                float sigma = 2.0f / (in);
                fprintf(stderr, "HeInit: sigma: %f\n", sigma);

                std::default_random_engine generator;
                std::normal_distribution<float> distribution(0,sigma);
                for (int i = 0; i < x.getWholeSize(); ++i)
                {
                    xData[i] = distribution(generator);
                    // fprintf(stderr, "xData[%d] = %f\n", i, xData[i]);
                }
            }
            else if (distri_ == Distribution::UNIFORM) {
                float sigma = 6.0f / (in);

                std::default_random_engine generator;
                std::uniform_real_distribution<float> distribution(0,sigma);
                for (int i = 0; i < x.getWholeSize(); ++i)
                {
                    xData[i] = distribution(generator);
                }
            }
        }
    };
} // namespace mkt



#endif
