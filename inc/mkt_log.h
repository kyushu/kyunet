#ifndef MKT_LOG_HPP
#define MKT_LOG_HPP

#include <stdarg.h>
#include <iostream>

namespace mkt {

    /*
        LOG_MOD Value:
            0x00: Turn off
            0x01: DISPLAY DEBUG MESSAGE
            0x02: DISPLAY DATA MESSAGE
     */
    #define LOG_MOD 0x02


    // C Style
    void inline mktLog(int log_mode, const char* format, ...)
    {
    #ifdef DEBUG_LOG

        int enable = log_mode & LOG_MOD;
        if (enable == 0)
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


    void inline MKT_ERR_LOG(const char* format, ...)
    {
        static char buf[1024];
        va_list args;
        va_start(args, format);
        vsprintf(buf, format, args);
        va_end(args);
        fprintf(stderr, "%s", buf);
    }


}


#ifdef MKTASSERT
#   define MKT_Assert(Expr, Msg) \
    __M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg);
#else
#   define MKT_Assert(Expr, Msg) ; // no value = do nothing
#endif

// static void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const std::string& msg);


template<typename T>
void CHECK_EQ(T a, T b, const char* func, int line) {
    // const char* file = __FILE__;
    // int line = __LINE__;
    if (a != b)
    {
        // std::cerr << a << "!=" << b << "\n"
        //     << "Source:\t\t" << file << ", line " << line << "\n";
        std::cerr << a << "!=" << b << "\n"
            << "Source:\t\t" << func <<  ", line: " << line << "\n";
        abort();
    }
}

template<typename T>
void CHECK_LT(T a, T b, const char* func, const int line) {
    // const char* file = __FILE__;
    // int line = __LINE__;
    if (a > b)
    {
        // std::cerr << a << ">" << b << "\n"
        //     << "Source:\t\t" << file << ", line " << line << "\n";
        std::cerr << a << ">" << b << "\n"
            << "Source:\t\t" << func << ", line: " << line << "\n";
        abort();
    }
}

template<typename T>
void CHECK_GE(T a, T b, const char* func, int line) {
    // const char* file = __FILE__;
    // int line = __LINE__;
    if (a < b)
    {
        // std::cerr << a << "<" << b << "\n"
        //     << "Source:\t\t" << file << ", line " << line << "\n";
        std::cerr << a << "<" << b << "\n"
            << "Source:\t\t" << func <<  ", line: " << line << "\n";
        abort();
    }
}



#endif
