#include <iostream>
#include <cstdio>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.hpp"
#include "folder_file_utils.hpp"

#include "tensor.h"
#include "layer.h"
#include "inputLayer.h"
#include "net.h"

using namespace mkt;

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

unsigned char* resize_image(std::string file, int out_w, int out_h) {

    // Load image from file
    int w, h, c;
    unsigned char *pImg = stbi_load(file.c_str(), &w, &h, &c, 0);

    if (w == out_w && h == out_h)
    {
        return pImg;
    }

    // unsigned char *r_img = (unsigned char*) malloc(out_w*out_h*c);
    unsigned char *r_img = new unsigned char[out_w*out_h*c];

    // resize 1
    // stbir_resize_uint8(pImg, w, h, 0, r_img, out_w, out_h, 0, c);

    // resize 2
    stbir_resize(pImg, w, h, 0, r_img, out_w, out_h, 0, STBIR_TYPE_UINT8, c, STBIR_ALPHA_CHANNEL_NONE, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR, NULL);

    return r_img;
}

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


    // Preset test data
    unsigned char s1[] = {1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3};
    unsigned char s2[] = {4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6, 4,5,6};
    unsigned char s3[] = {7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 7,8,9};

    int channel = 3;
    int height = 4;
    int width = 3;
    int batchSize = 3;


    // Configure, initialize network
    KyuNet net;
    net.addInputLayer("input", batchSize, height, width, channel);
    net.Compile();

    // Get InputLayer
    InputLayer* pInput = net.InputLayer();
    const float *pdata = pInput->pDst_->getCPUData();
    int size3D = pInput->pDst_->getSize3D();
    fprintf(stderr, "size3D: %d\n", size3D);

    // Add data
    pInput->FlattenImageToTensor(s1, true);
    pInput->FlattenImageToTensor(s2, true);
    pInput->FlattenImageToTensor(s3, true);

    // Verify data
    int wholeSize = pInput->pDst_->WholdSize();
    for (int i = 0; i < wholeSize; ++i)
    {
        printf("%d - %f\n", i, pInput->pDst_->pData_[i] * 255);
    }

}

void test_add_batch_image() {


    std::string img_dir = "../example/images";
    std::vector<std::string> file_list;
    mkt::listdir(img_dir.c_str(), file_list);
    fprintf(stderr, "file_list.size(): %ld\n", file_list.size());
    for (int i = 0; i < file_list.size(); ++i)
    {
        file_list.at(i) = img_dir + "/" + file_list.at(i);
    }

    // std::vector<std::string>::const_iterator first = file_list.begin();
    // std::vector<std::string>::const_iterator last = file_list.begin() + 3;
    // std::vector<std::string> sub_file_list(first, last);

    int height = 600;
    int width = 480;
    int channel = 3;
    int batchSize = 3;

    int num_iter = file_list.size() / batchSize;


    KyuNet net;
    /***********************************************
     *  Step 1. Configure KyuNetwork
     *  Add layer by parameters
     **********************************************/
    net.addInputLayer("input", batchSize, height, width, channel);


    /***********************************************
     *  Step 2. Initialize KyuNetwork
     *  Initialize Tensor by updated parameters
     **********************************************/
    net.Compile();


    /**********************************************
     *  Step 3. Add batch data
     *********************************************/
    for (int i = 0; i < 1; ++i)
    {
        int start = i * batchSize;
        int end = (i+1) * batchSize;
        std::vector<std::string>::const_iterator first = file_list.begin() + start;
        std::vector<std::string>::const_iterator last = file_list.begin() + end;
        std::vector<std::string> batchFile(first, last);
        net.add_data_from_file_list(batchFile);
    }


    /***********************************************
     * Verify
     **********************************************/
    InputLayer* pInput = net.InputLayer();
    const float *pdata = pInput->pDst_->getCPUData();
    int size3D = pInput->pDst_->getSize3D();
    fprintf(stderr, "size3D: %d\n", size3D);

    unsigned char* pImg = new unsigned char[size3D];
    for (int i = 0; i < batchSize; ++i)
    {
        pInput->DeFlattenImage(pdata, height, width, channel, pImg);

        if (channel == 3)
        {
            cv::Mat img_mat(height, width, CV_8UC3, pImg);
            cv::cvtColor(img_mat, img_mat, CV_RGB2BGR);
            printf("img_mat.col: %d, row: %d\n", img_mat.cols, img_mat.rows);
            cv::imshow("image", img_mat);
            cv::waitKey(0);
        }
        else if(channel == 1)
        {
            cv::Mat img_mat(height, width, CV_8UC1, pImg);
            // printf("img_mat.col: %d, row: %d\n", img_mat.cols, img_mat.rows);
            cv::imshow("image", img_mat);
            cv::waitKey(0);
        }

        pdata += size3D;

    }
}

int main(int argc, char const *argv[])
{


    // test_add_batch_image();
    test_flatten_image();


    return 0;
}
