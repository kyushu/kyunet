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

#ifndef MKT_DEFINITIONS_h
#define MKT_DEFINITIONS_h

#include <vector>

namespace mkt {

    enum class OP_STATUS: char
    {
        SUCCESS = 0,
        FAIL    = 1,
        OVER_MAX_SIZE = 3,
        UNMATCHED_SIZE = 4,

    };

    // Net Type
    enum class NetMode: int
    {
        TRAINING=0,
        INFER
    };
    // Layer Type
    enum class LayerType:int
    {
        INPUT = 0,
        DENSE,
        CONVOLUTION,
        POOLING,
        RELU,
        SIGMOID,
        SOFTMAX,
        CROSS_LOSS_SOFTMAX
    };

    // Weight Initializer type
    enum class InitializerType: int
    {
        NONE=0,
        ONE,
        TEST, /*TEST*/
        ZERO,
        XAVIER_NORM,
        XAVIER_UNIFORM,
        HE_INIT_NORM,
        HE_INIT_UNIFORM

    };

    // value Distribution type
    enum class Distribution: int
    {
        UNIFORM=0,
        NORM
    };

    // Activattion type
    enum class ActivationType: int
    {
        NONE=0,
        SIGMOID,
        //TANH,
        RELU,
        LRELU
        //SELU

    };

    // Pooling type
    enum class PoolingMethodType: int
    {
        MAX=0,
        AVG
    };

    // Loss function type
    enum class LossType: int
    {
        CROSS_ENTROPY_LOSS = 0

    };


    // Padding type
    /*
        "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order. See WaveNet: A Generative Model for Raw Audio, section 2.1.
     */
    enum class PaddingType:int
    {
        VALID=0, /*No padding*/
        SAME     /*padding the input such that the output has the same length as the original input*/
        // causal
    };

    enum CBLAS_TRANSPOSE
    {
        CblasNoTrans=0,
        CblasTrans
    };

    enum class RegularizationType:int
    {
        L1,
        L2
    };

    typedef std::vector<int> Shape;
} // namespace mkt



#endif
