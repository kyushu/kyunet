

#include <random>

int main(int argc, char const *argv[])
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0, 1.0f);

     for (int i=0; i<100; ++i) {
        float number = distribution(generator);
        fprintf(stderr, "number: %f\n", number);
    }

    return 0;
}
