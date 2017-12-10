#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <chrono>


namespace mkt {

    static  bool caseInsensitiveCompare(const std::string a, const std::string b)
    {
        unsigned int sz = a.size();

        if (b.size() != sz) {
            return false;
        }

        for (unsigned int i = 0; i < sz; ++i) {
            if (std::tolower(a[i]) != std::tolower(b[i])) {
                return false;
            }
        }
        return true;
    }


    /*
     * Return a string contains year, month, date
     */
    static std::string getNowDateString() {

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y");
        auto timeString = oss.str();
        return oss.str();
    }

    /*
     * Return a string contains year, month, date, hour, min, sec
     */
    static std::string getFullNowDateString() {

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
        auto timeString = oss.str();
        return oss.str();
    }

    /*
     * Split string by specific token
     * text       : source string
     * separator  : character for split string
     * return     : container contains separated strings  
     */
    static std::vector<std::string> split(const std::string &text, char separator) {
        std::vector<std::string> tokens;
        size_t start=0, end=0;
        while((end = text.find(separator, start)) != std::string::npos) {
            if (end != start) {
                tokens.push_back(text.substr(start, end - start));
            }
            start = end + 1;
        }

        if (end != start) {
            tokens.push_back(text.substr(start));
        }
        return tokens;
    }

    bool has_only_digits(const std::string s){
      return s.find_first_not_of( "0123456789" ) == std::string::npos;
    }
}


#endif