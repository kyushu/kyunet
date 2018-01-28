#ifndef _MKT_LOG_HPP_
#define _MKT_LOG_HPP_

#include <stdarg.h>

namespace {

    /*
        LOG_MOD Value:
            0x00: DISPLAY ALL MESSAGE
            0x01: DISPLAY DEBUG MESSAGE
            0x02: DISPLAY DATA MESSAGE
     */
    #define LOG_MOD 0x02


    // C Style
    static void inline mktLog(int log_mode, const char* format, ...)
    {
    #ifdef DEBUG_LOG


        int enable = log_mode & LOG_MOD;
        if (LOG_MOD > 0 && enable == 0)
        {
            return;
        }

        static char buf[1024];
        va_list args;
        va_start(args, format);
        vsprintf(buf, format, args);
        va_end(args);
        fprintf(stderr, "%s", buf);



    #endif
    }

}

#endif
