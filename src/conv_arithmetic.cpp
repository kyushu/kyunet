
#include "conv_arithmetic.h"
#include <math.h>
#include "mkt_log.h"

namespace mkt{
namespace conv{

void calcOutputSize (
        const int& inputW, const int& inputH, const int& filterW, const int& filterH,
        ConvParam& param, int& ootputW, int& outputH )
    {
        switch(param.paddingType_) {
            case PaddingType::VALID:
                /**
                 * The definition of VALID is
                 * stride_w = stride_h = 1
                 * pad_w = 0
                 * pad_h = 0
                 */
                param.stride_w_ = 1;
                param.stride_h_ = 1;
                param.pad_w_ = 0;
                param.pad_h_ = 0;
                ootputW = static_cast<int>( static_cast<float>(inputW - filterW + 1) );
                outputH = static_cast<int>( static_cast<float>(inputH - filterH + 1) );
                break;

            case PaddingType::SAME:

                /**
                 * The definition of SAME(Half) is
                 * stride_w = stride_h = 1
                 * fw = 2n+1
                 * fh = 2n+1
                 * pad_w = floor(fw/2)
                 * paf_h = floor(fh/2)
                 */
                MKT_Assert(filterW%2 == 1, "fw != 2n+1");
                MKT_Assert(filterH%2 == 1, "fh != 2n+1");

                ootputW = inputW;
                outputH = inputH;
                param.stride_w_ = 1;
                param.stride_h_ = 1;
                param.pad_w_ = floor( static_cast<float>(filterW)/2 );
                param.pad_h_ = floor( static_cast<float>(filterH)/2 );

                MKT_Assert(ootputW = inputW, "iw != ow");
                MKT_Assert(outputH = inputH, "ih != oh");


                break;

            case PaddingType::FULL:
                /**
                 * The definition of FULL is
                 * stride_w = stride_h = 1
                 * pad_w = fw - 1
                 * paf_h = fh - 1
                 */
                param.stride_w_ = 1;
                param.stride_h_ = 1;
                param.pad_w_ = filterW - 1;
                param.pad_h_ = filterH - 1;

                ootputW = inputW + (filterW - 1);
                outputH = inputH + (filterH - 1);

                break;

            case PaddingType::NORMAL:
                ootputW = static_cast<int>( static_cast<float>(inputW - filterW + 2*param.pad_w_) / param.stride_w_ ) + 1;
                outputH = static_cast<int>( static_cast<float>(inputH - filterH + 2*param.pad_h_) / param.stride_h_ ) + 1;
                break;
            default:
                ootputW = static_cast<int>( static_cast<float>(inputW - filterW + 2*param.pad_w_) / param.stride_w_ ) + 1;
                outputH = static_cast<int>( static_cast<float>(inputH - filterH + 2*param.pad_h_) / param.stride_h_ ) + 1;
                break;
        }
    }

} // namespace conv
} // namespace mkt
