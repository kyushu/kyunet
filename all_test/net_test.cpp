#include <iostream>

#include "utils.hpp"
#include "folder_file_utils.hpp"

#include "tensor.h"
#include "net.h"

/*
    To use stb you must define 
        #define STB_IMAGE_IMPLEMENTATION         // for load image
        #define STB_IMAGE_RESIZE_IMPLEMENTATION  // for resize image
    to enable functions.
    but i already define these keyword in inc_thirdparty.h
    so we don't need to define again
*/ 
#include "stb_image.h"
#include "stb_image_resize.h"



void test_flatten_image() {
    /**
        Test reord RGB pixel from {rgb, rgb, ... ,rgb} to {rrrr..., gggg..., bbbb...}
    */
    
    /*
      the following code will normalize each pixel of image from (0, 255)  to (-1, 1)
      and the pixel order will be reordered
      assume image is 3x4x3 = wxhxc
      the original order of pixel of image is 
      
      0 1 2 3 4 5 6 7 8
      R G B R G B R G B
      R G B R G B R G B
      R G B R G B R G B
      R G B R G B R G B

      and convert it into 

      0 1 2 3 4 5 6 7 8
      R R R R R R R R R
      R R R G G G G G G
      G G G G G G B B B
      B B B B B B B B B

    */ 
    unsigned char s1[] = {1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3};
    unsigned char s2[] = {4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6};
    unsigned char s3[] = {7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9};

    int depth = 3;
    int height = 4; 
    int width = 3;

    mkt::Net<float> net = mkt::Net<float>();
    net.pInput = new mkt::Tensor<float>(width, height, depth);
    net.pInput->initTensorWithBatchSize(3);

    net.flattenImage(s1, true);
    net.pInput->data_wr_idx_++;
    net.flattenImage(s2, true);
    net.pInput->data_wr_idx_++;
    net.flattenImage(s3, true);
    net.pInput->data_wr_idx_++;
    for (int i = 0; i < 36*3; ++i)
    {
      printf("%d - %f\n", i, net.pInput->pData_[i]);
    }
}

int main(int argc, char const *argv[])
{

    // std::string img_dir = "../../example/images/";
    // std::vector<std::string> file_list;
    // mkt::listdir(img_dir.c_str(), file_list);

    // mkt::Net<float> net = mkt::Net<float>();

    // std::string file = img_dir + file_list.at(0);
    // printf("%s\n", file.c_str());

    // // get width, height and number of channel of image
    // int w, h, c;
    // unsigned char *pImg = stbi_load(file.c_str(), &w, &h, &c, 0);
    // printf("image width: %d, height: %d, channel: %d\n", w, h, c);

    // net.pInTensor = new mkt::Tensor<float>(file_list.size(), w, h, c);

    // // Load image from file and normalize pixel value in range 0 to 1
    // int depth = net.pInPutTensor->getDepth();
    // int height = net.pInPutTensor->getHeight();
    // int width = net.pInPutTensor->getWidth();

    
    test_flatten_image();
    



    return 0;
}