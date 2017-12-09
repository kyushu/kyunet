#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


int main(int argc, char const *argv[])
{
    
    std::string img_file = "../../example/images/Marion-Cotillard1.jpg";

    printf("hellow kyunet\n");
    int w, h, c;
    unsigned char *arr_img = stbi_load(img_file.c_str(), &w, &h, &c, 0);
    printf("w: %d, h: %d, c: %d\n", w, h, c);

    // BGR: CV_8UV3
    // Gray-scale: CV_8UC1
    cv::Mat img_mat(h,w,CV_8UC3,arr_img);
    cv::cvtColor(img_mat, img_mat, CV_RGB2BGR);

    printf("img_mat.col: %d, row: %d\n", img_mat.cols, img_mat.rows);
    cv::imshow("image", img_mat);
    cv::waitKey(0);
    
    return 0;
}