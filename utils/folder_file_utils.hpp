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
        
        std::vector<std::string> path_list = split(target_folder, '/');
        std::string path = "";
        for (int i = 0; i < path_list.size(); ++i)
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

        std::vector<std::string> path_list = split(target_folder, '/');
        std::string path = "";
        for (int i = 0; i < path_list.size(); ++i)
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