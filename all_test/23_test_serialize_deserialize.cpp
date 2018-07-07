

#include <fstream>
#include <vector>

#include "test_utils.h"

#include "tensor.h"


void test_write()
{
    std::vector<float> vec1 = {3, 4, 5, 6,
                        7, 4, 9, 1,
                        5, 2, 5, 1,
                        8, 1, 1, 4,
                        1, 2, 2, 3,
                        3, 1, 1, 4};
    // vector<float> vec1(testData1, testData1 + sizeof(testData1) / sizeof(testData1[0]) );

    std::vector<float> vec2 = {1, 3, 5, 7,
                        9, 11, 13, 15,
                        2, 4, 6, 8,
                        10, 12, 14, 16,
                        11, 22, 33, 44,
                        55, 66, 77, 88};
    // vector<float> vec2(testData2, testData2 + sizeof(testData2) / sizeof(testData2[0]) );


    mkt::Tensor<float>* pTensor1 = new mkt::Tensor<float>{2, 1, 1, 12};
    pTensor1->allocate();
    pTensor1->addData(vec1);

    mkt::Tensor<float>* pTensor2 = new mkt::Tensor<float>{1, 2, 2, 6};
    pTensor2->allocate();
    pTensor2->addData(vec2);

    std::fstream file{"./test_save.bin", std::ios::out | std::ios:: binary};



    pTensor1->serialize(file, false);
    pTensor2->serialize(file, false);
}

int main(int argc, char const *argv[])
{

    test_write();

    return 0;
}
