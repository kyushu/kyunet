#include <iostream>

#include "utils.hpp"
#include "folder_file_utils.hpp"

#include "net.h"

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


int main(int argc, char const *argv[])
{

    std::string img_dir = "../../example/images/";
    std::vector<std::string> file_list;
    mkt::listdir(img_dir.c_str(), file_list);

    mkt::Net net = mkt::Net();

    std::string file = img_dir + file_list.at(0);
    printf("%s\n", file.c_str());

    // get width, height and number of channel of image
    int w, h, c;
    unsigned char *pImg = stbi_load(file.c_str(), &w, &h, &c, 0);
    printf("image width: %d, height: %d, channel: %d\n", w, h, c);

    net.pInTensor = new mkt::Tensor(file_list.size(), w, h, c);

    // Load image from file and normalize pixel value in range 0 to 1
    int depth = net.pInTensor->getDepth();
    int height = net.pInTensor->getHeight();
    int width = net.pInTensor->getWidth();

    /*
      the following code will normalize each pixel of image from (0, 255)  to (-1, 1)
      and the reording the pixel order
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
    
    for (int i = 0; i < file_list.size(); ++i)
    {

        std::string img_file = img_dir + file_list.at(0);
        pImg = stbi_load(img_file.c_str(), &w, &h, &c, 0);

        for (int c = 0; c < depth; ++c)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    net.pInTensor->data_[width * (height * c + y) + x] = ((float)pImg[depth * (x + width * y) + c] / 255.0f -0.5f) * 2.0f;
                }
            }
        }

        net.pInTensor->data_wr_idx++;
    }


    return 0;
}