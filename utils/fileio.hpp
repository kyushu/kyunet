#ifndef _FILE_IO_H_
#define _FILE_IO_H_

#include <fstream>

#include "mkt_log.h"

namespace mkt {

    class FileIO
    {
        std::fstream filestream;
    public:
        FileIO(std::string pathName, bool bWrite)
        {
            if (bWrite)
            {
                filestream.open(pathName, std::ios::out | std::ios::binary);
            } else {
                filestream.open(pathName, std::ios::in | std::ios::binary);
            }
        }
        ~FileIO();
        
        bool isOpen()
        {
            return filestream.is_open();
        }

        /**
         * Write to file
         */
        void write(const int i)
        {
            if (filestream.good())
            {
                filestream << i;
            } else {
                MKT_Assert(filestream.good() != true, "filestream is not good");
            }
            
        }

        void write(const float i)
        {
            if (filestream.good())
            {
                filestream << i;
            } else {
                MKT_Assert(filestream.good() != true, "filestream is not good");
            }
        }

        /**
         * Read from file
         */
        void read(int& i)
        {
            if (filestream.good)
            {
                filestream >> i;
            } else {
                MKT_Assert(filestream.good() != true, "filestream is not good");
            }
        }

        void read(float& i)
        {
            if (filestream.good)
            {
                filestream >> i;
            } else {
                MKT_Assert(filestream.good() != true, "filestream is not good");
            }
        }
    };

}


#endif
