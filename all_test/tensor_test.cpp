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


int main(int argc, char const *argv[])
{
    
    std::string img_dir = "../../example/images/";

    std::vector<std::string> file_list;
    mkt::listdir(img_dir.c_str(), file_list);


    mkt::Tensor tensor = mkt::Tensor(3, 480, 600, file_list.size());
    printf("batch size: %d\n", tensor.getBatchSize());
    printf("width: %d\n", tensor.getWidth());
    printf("height: %d\n", tensor.getHeight());
    printf("channel: %d\n", tensor.getDepth());


    for (int i = 0; i < file_list.size(); ++i)
    {
        std::string img_file = img_dir + file_list.at(i);
        int ori_w, ori_h, ori_c;
        int out_w = 100;
        int out_h = 100;
        // unsigned char *img = stbi_load(img_file.c_str(), &ori_w, &ori_h, &ori_c, 0);
        
        tensor.addData(img_file.c_str());
        // unsigned char *r_img = (unsigned char*) malloc(out_w*out_h*ori_c);

        // resize 1
        // stbir_resize_uint8(img, ori_w, ori_h, 0, r_img, out_w, out_h, 0, ori_c);

        // resize 2
        // stbir_resize(img, ori_w, ori_h, 0, r_img, out_w, out_h, 0, STBIR_TYPE_UINT8, ori_c, STBIR_ALPHA_CHANNEL_NONE, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR, NULL);
        
    }

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



    
    
    return 0;
}