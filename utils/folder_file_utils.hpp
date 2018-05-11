/*
* Copyright (c) 2017 Morpheus Tsai.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*
    Only LINUX is supported.
 */

#ifndef _FOLDER_FILE_UTILS_H_
#define _FOLDER_FILE_UTILS_H_

#include <iostream>
#include <dirent.h>
#include <sys/stat.h>   // create directory
#include "utils.hpp"

namespace mkt {
    static int listdir(const char *path, std::vector<std::string> &file_list) {
        struct dirent *entry;
        DIR *dp;

        dp = opendir(path);
        if (dp == NULL) {
            perror("opendir: Path does not exist or could not be read.");
            return -1;
        }

        while ((entry = readdir(dp))){
            std::string fname(entry->d_name);
            // puts(entry->d_name);
            if (fname != "." && fname != "..")
            {
                file_list.push_back(entry->d_name);
            }
        }

        closedir(dp);
        return 0;
    }

    static bool checkFolderExist (std::string target_folder) {

        struct stat sb;
        if (stat(target_folder.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
        {
            // printf("%s exists\n", target_folder.c_str());
            return true;
        }
        else
        {
            printf("%s does not exist\n", target_folder.c_str());
            return false;
        }
    }
    static bool checkFolderRecursive(std::string target_folder) {

        std::vector<std::string> path_list = UTILS::split(target_folder, '/');
        std::string path = "";
        for (size_t i = 0; i < path_list.size(); ++i)
        {
            path = path + path_list.at(i) + '/';
            if (!checkFolderExist(path))
            {
                printf("%s does not exist\n", path.c_str());
                return false;
            } else {
                printf("%s exists\n", path.c_str());
            }
        }

        return true;
    }

    static bool createFolder(std::string target_folder) {

        printf("create target_folder: %s\n", target_folder.c_str());

        const int dir_err = mkdir(target_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        // const int dir_err = mkdir(target_folder.c_str(), 755);
        if (-1 == dir_err)
        {
            printf("Error creating directory!\n");
            return false;
        } else {
            return true;
        }
    }

    static bool createFolderRecursive(std::string target_folder) {

        std::vector<std::string> path_list = UTILS::split(target_folder, '/');
        std::string path = "";
        for (size_t i = 0; i < path_list.size(); ++i)
        {
            path = path + path_list.at(i) + '/';
            if (!checkFolderExist(path))
            {
                if (!createFolder(path)) {
                    printf("create %s faile\n", path.c_str());
                    return false;
                } else {
                    printf("%s is created\n", path.c_str());
                }
            } else {
                printf("%s is exist\n", path.c_str());
            }
        }

        return true;
    }
}



#endif
