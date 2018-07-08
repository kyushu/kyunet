

#include <fstream>
#include <vector>

#include "test_utils.h"

#include "tensor.h"


void test_write(std::string file_path)
{
    // Prepare test data of Tensor 1
    std::vector<float> vec1 = {3, 4, 5, 6,
                        7, 4, 9, 1,
                        5, 2, 5, 1,
                        8, 1, 1, 4,
                        1, 2, 2, 3,
                        3, 1, 1, 4};

    // Prepare test data of Tensor 2
    std::vector<float> vec2 = {1, 3, 5, 7,
                        9, 11, 13, 15,
                        2, 4, 6, 8,
                        10, 12, 14, 16,
                        11, 22, 33, 44,
                        55, 66, 77, 88};

    // Create, allocate and add data of first Tensor
    mkt::Tensor<float>* pTensor1 = new mkt::Tensor<float>{2, 1, 1, 12};
    pTensor1->allocate();
    pTensor1->addData(vec1);
    mkt::print_matrix(2, 1, 1, 12, pTensor1->getCPUData());

    // Create, allocate and add data of first Tensor
    mkt::Tensor<float>* pTensor2 = new mkt::Tensor<float>{1, 2, 2, 6};
    pTensor2->allocate();
    pTensor2->addData(vec2);
    mkt::print_matrix(1, 2, 2, 6, pTensor2->getCPUData());


    // Open file with fstream
    std::fstream file{file_path, std::ios::out | std::ios:: binary};

    // Write Tensor1 data to file
    pTensor1->serialize(file, false);
    // Write Tensor2 data to file
    pTensor2->serialize(file, false);

    file.close();
}

void test_read(std::string file_path)
{
    printf("### Start read data from file ###\n");

    /*-
     * Because the binary data include data of two tensor
     * we create two tensor corresponding to the test_write function
     */
    // Create fisrt tensor to load data from file
    mkt::Tensor<float>* pTensor1 = new mkt::Tensor<float>{2, 1, 1, 12};
    pTensor1->allocate();

    // Create second tensor to load data from file
    mkt::Tensor<float>* pTensor2 = new mkt::Tensor<float>{1, 2, 2, 6};
    pTensor2->allocate();

    // Open file with fstream
    std::fstream file{file_path, std::ios::in | std::ios:: binary};

    // Get the initial location of fstream get-pointer
    int pos = file.tellg();
    printf("begin pos: %d\n", pos);

    // Read Tensor 1 data from fstream
    pTensor1->deserialize(file, true);

    // Get current location of fstream get-pointer
    pos = file.tellg();
    printf("after read tensor1 pos: %d\n", pos);

    // Read Tensor 2 data from fstream
    pTensor2->deserialize(file, false);

    // Get current location of fstream get-pointer
    pos = file.tellg();
    printf("after read tensor2 pos: %d\n", pos);


    mkt::print_matrix(2, 1, 1, 12, pTensor1->getCPUData());
    mkt::print_matrix(1, 2, 2, 6, pTensor2->getCPUData());

    file.close();
}


int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        printf("%s [Mode] dst_file\n", argv[0]);
        printf("Mode = 0: write\n");
        printf("Mode = 1: read\n");
        printf("dst_file: given a file path\n");
        return -1;
    }

    std::string mode = argv[1];
    std::string file_path = argv[2];

    printf("mode: %s\n", mode.c_str());
    printf("file_path: %s\n", file_path.c_str());

    if (mode == "0")
    {
        test_write(file_path);
    } else {
        test_read(file_path);
    }

    return 0;
}
