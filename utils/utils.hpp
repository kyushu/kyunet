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

    namespace UTILS {
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

    } // namespace UTILS
} // namespace mkt


#endif
