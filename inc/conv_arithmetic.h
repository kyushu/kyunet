
#ifndef CONV_ARITHEMTIC_HPP
#define CONV_ARITHEMTIC_HPP

#include "definitions.h"
#include "params.h"

namespace mkt{
namespace conv{


    void calcOutputSize (
        const int& inputW, const int& inputH, const int& filterW, const int& filterH,
        ConvParam& param, int& ootputW, int& outputH );


} // namespace conv
} // namespace mkt

#endif
