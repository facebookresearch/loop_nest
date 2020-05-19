#include "loop_nest_bench.h"
#include "loop_nest_baseline.h"

#include <xmmintrin.h>

#ifndef CT_ISA
#define CT_ISA avx2
#endif

int main()
{
    // DAZ
    _mm_setcsr(_mm_getcsr() | 0x0040);
    // FTZ
    _mm_setcsr(_mm_getcsr() | 0x8000);

    using facebook::sysml::aot::avx2;
    using facebook::sysml::aot::avx2_plus;
    using facebook::sysml::aot::avx512;

    using facebook::sysml::aot::loop_nest_bench;
    // 2D convolution NCHW example:
    // O(c_out, o_h, o_w) = I(c_i, o_h + k_h, ow + k_w) * K(c_o, c_i,
    // k_h, k_w)
    {
        int CIN  = 64;
        int COUT = 64;
        int OS   = 112;
        int KS   = 3;
        int IS   = OS + KS - 1;

        loop_nest_bench<CT_ISA>(
            {{"c_out", 16}, //
             {"o_h", 1},
             {"o_w", 28},
             {"c_in", 16},
             {"c_in", 1},
             {"c_out", 1}, //
             {"o_w", 1},   //
             {"k_h", 1},   //
             {"k_w", 1}},  //
            // The second argument is a map of the dimension sizes
            {{"c_out", COUT},
             {"o_w", OS},
             {"k_w", KS},
             {"c_in", CIN},
             {"o_h", OS},
             {"k_h", KS},
             {"i_w", IS},
             {"i_h", IS}},
            // Vars of C (other variables are reduction variables)
            {"c_out", "o_w", "o_h"},
            // Variables of A, note that i_w and i_h are not used
            {"c_in", "i_w", "i_h"},
            // Variables of B
            {"c_in", "c_out", "k_w", "k_h"},
            // C's strides for each variable
            {{"o_w", 1}, {"c_out", OS * OS}, {"o_h", OS}},
            // A's strides for each variable Note how we
            // provide strides for i/k_h and i/k_w, this is
            // because the access to A is based on output
            // and reduction variables
            {{"o_w", 1},
             {"k_w", 1},
             {"c_in", IS * IS},
             {"o_h", IS},
             {"k_h", IS}},
            // B's strides for each variable
            {{"c_out", KS * KS * CIN},
             {"c_in", KS * KS},
             {"k_w", 1},
             {"k_h", KS}},
            64);
    }

    // 2D convolution on NCHW16c layout example:
    // O(g_out, c_out, o_h, o_w) = I(g_in, c_in, o_h + k_h, ow + k_w) *
    //                             K(g_in, g_out, c_in, c_out, k_h, k_w)
    // if (0)
    {
        int GIN  = 128 / 16;
        int CIN  = 16;
        int GOUT = 128 / 16;
        int COUT = 16;
        int OS   = 56;
        int KS   = 3;
        int IS   = OS + KS - 1;

        loop_nest_bench<CT_ISA>(
            {{"g_out", 1}, //
             {"o_w", 28},
             {"o_h", 1},
             {"g_in", 1},
             {"c_in", 1},
             {"o_w", 1}, //
             //{"o_w", 1},    //
             {"k_h", 1},    //
             {"k_w", 1},    //
             {"c_out", 1}}, //
            // The second argument is a map of the dimension sizes
            {{"g_out", GOUT},
             {"c_out", COUT},
             {"o_w", OS},
             {"k_w", KS},
             {"g_in", GIN},
             {"c_in", CIN},
             {"o_h", OS},
             {"k_h", KS},
             {"i_h", IS},
             {"i_w", IS}},
            // Vars of C (other variables are reduction variables)
            {"g_out", "c_out", "o_w", "o_h"},
            // Variables of A, note that i_w and i_h are not used
            {"g_in", "c_in", "i_w", "i_h"},
            // Variables of B
            {"g_out", "g_in", "c_in", "c_out", "k_w", "k_h"},
            // C's strides for each variable
            {{"g_out", OS * OS * COUT},
             {"o_h", OS * COUT},
             {"o_w", COUT},
             {"c_out", 1}},
            // A's strides for each variable Note how we
            // provide strides for i/k_h and i/k_w, this is
            // because the access to A is based on output
            // and reduction variables
            {{"g_in", IS * IS * CIN},
             {"o_h", IS * CIN},
             {"k_h", IS * CIN},
             {"o_w", CIN},
             {"k_w", CIN},
             {"c_in", 1}},
            // B's strides for each variable
            {{"g_in", COUT * KS * KS * CIN * GOUT},
             {"g_out", COUT * KS * KS * CIN},
             {"c_in", COUT * KS * KS},
             {"k_h", COUT * KS},
             {"k_w", COUT},
             {"c_out", 1}});
    }
}
