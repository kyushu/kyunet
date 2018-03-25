#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

namespace mkt {

    static void print_matrix(int batchSize, int channel, int height, int width, float* pData) {
        int size2D = height*width;
        int size3D = height*width*channel;
        fprintf(stderr, "batchSize: %d\tchannel:%d\theight:%d\twidth:%d\t\n", batchSize, channel, height, width);
        for (int b = 0; b < batchSize; ++b)
        {
            fprintf(stderr, "batch: %d\n", b);
            for (int c = 0; c < channel; ++c)
            {
                fprintf(stderr, "channel: %d\n", c);
                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        fprintf(stderr, "%.3f\t", pData[w + h*width + c*size2D + b*size3D]);
                    }
                    if (height > 1) { fprintf(stderr, "\n"); }
                }
                if (channel > 1) { fprintf(stderr, "\n"); }
            }
            if (batchSize > 1) { fprintf(stderr, "\n"); }
        }

        fprintf(stderr, "\n");
    }

    static void genRndPseudoData(float* pData, int num, int ch, int height, int width) {

        for (int b = 0; b < num; ++b)
        {
            for (int c = 0; c < ch; ++c)
            {
                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        float rndVal = (rand() % 1000) / 1000.0f;
                        pData[w + h*width + c*(height*width) + b*(ch*height*width)] = rndVal;
                        // fprintf(stderr, "data[%d]=%f\n", w + h*width + c*(height*width) + b*(channel*height*width), rndVal);
                    }
                }
            }
        }
    }
}

#endif
