#include <iostream>

#include "net.h"

int main(int argc, char const *argv[])
{

    std::string img_dir = "../../example/images/";
    std::vector<std::string> file_list;
    mkt::listdir(img_dir.c_str(), file_list);




    return 0;
}