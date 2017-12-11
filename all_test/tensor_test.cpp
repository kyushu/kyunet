#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// #define STB_IMAGE_IMPLEMENTATION
// #include "stb/stb_image.h"
// #define STB_IMAGE_RESIZE_IMPLEMENTATION
// #include "stb/stb_image_resize.h"

#include "utils.hpp"
#include "folder_file_utils.hpp"

#include "tensor.h"

/*
  To use stb you must define 
  #define STB_IMAGE_IMPLEMENTATION         // for load image
  #define STB_IMAGE_RESIZE_IMPLEMENTATION  // for resize image
  to enable functions.
  but i already define these keyword in libkyunet and linked
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

void test_load_image_file(std::string img_dir) {

    std::vector<std::string> file_list;
    mkt::listdir(img_dir.c_str(), file_list);

    mkt::Tensor tensor = mkt::Tensor(file_list.size(), 480, 600, 3);
    printf("batch size: %d\n", tensor.getBatchSize());
    printf("width: %d\n", tensor.getWidth());
    printf("height: %d\n", tensor.getHeight());
    printf("channel: %d\n", tensor.getDepth());

    assert(file_list.size() >= 3);

    /*********************
     * Add data to tensor
     ********************/
    // add from file
    std::string img_file = img_dir + file_list.at(0);
    tensor.addData(img_file.c_str());

    // add from float array
    img_file = img_dir + file_list.at(1);
    int w, h, c;
    unsigned char *pImg = stbi_load(img_file.c_str(), &w, &h, &c, 0);
    float* pfImg = new float[w*h*c];
    for (int i = 0; i < w*h*c; ++i)
    {
        *(pfImg+i) = (float)*(pImg+i);
    }
    tensor.addData(pfImg);

    // add from vector
    img_file = img_dir + file_list.at(2);
    pImg = stbi_load(img_file.c_str(), &w, &h, &c, 0);
    std::vector<float> vfImg;
    for (int i = 0; i < w*h*c; ++i)
    {
        vfImg.push_back( (float)*(pImg+i) );
    }
    tensor.addData(vfImg);


    /***************
     * Verify data 
     **************/
    const float *data = tensor.getData();
    unsigned char *uchar = (unsigned char *)calloc(tensor.getSize(), sizeof(unsigned char));
    for (int i = 0; i < file_list.size(); ++i)
    {
        // Display 
        // BGR: CV_8UV3
        // Gray-scale: CV_8UC1
        
        
        printf("read: %p\n", data);

        for (int i = 0; i < tensor.getSize(); ++i)
        {
            *(uchar+i) = int(*(data+i));
        }

        cv::Mat img_mat(600, 480, CV_8UC3, uchar);
        cv::cvtColor(img_mat, img_mat, CV_RGB2BGR);

        printf("img_mat.col: %d, row: %d\n", img_mat.cols, img_mat.rows);
        cv::imshow("image", img_mat);
        cv::waitKey(0);

        data += tensor.getSize();
    }
}

int main(int argc, char const *argv[])
{
    
    

    /****************************************** 
        Test load 3 image file from a directory 
    ******************************************/
    // std::string img_dir = "../../example/images/";
    // test_load_image_file(img_dir);

    /**
        Test reord RGB pixel 
    */
    float s[] = {1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3};
    float t[36];
    int depth = 3;
    int height = 4; 
    int width = 3;

    int sz = height*width;
    for (int i = 0; i < depth*height*width; i+=depth)
    {
        int idx = int(i/depth);
        t[idx] = s[i];
        t[sz*(depth-2) + idx] = s[i+1];
        t[sz*(depth-1) + idx] = s[i+2];
    }
    
    for (int i = 0; i < 36; ++i)
    {
        printf("%d - %f\n", i, *(t+i));
    }

    return 0;
}