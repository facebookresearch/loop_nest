// #include "dabun/thread/operating_cpu_set.hpp"

#include "baseline/loop_nest_baseline.hpp"
#include "dabun/arithmetic_operation.hpp"
#include "dabun/check.hpp"
#include "dabun/loop_nest.hpp"
#include "dabun/loop_nest_descriptor.hpp"
#include "dabun/random_vector.hpp"

#include <catch2/catch.hpp>

#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dabun::tests::baseline
{

template <class ISA>
void loop_nest_against_slow_baseline(
    std::vector<std::pair<std::string, int>> const& order,
    std::map<std::string, int> const&               sizes,
    std::set<std::string> const&                    C_formula,
    std::set<std::string> const&                    A_formula,
    std::set<std::string> const&                    B_formula,
    std::map<std::string, int> const&               C_strides,
    std::map<std::string, int> const&               A_strides,
    std::map<std::string, int> const& B_strides, int max_unrolled_fmas = 512,
    int alpha = 1)
{
    loop_nest_descriptor args(order, sizes, C_formula, A_formula, B_formula,
                              C_strides, A_strides, B_strides);

    auto req_sizes = args.get_required_sizes();

    std::int64_t C_size = req_sizes.C_size;
    std::int64_t A_size = req_sizes.A_size;
    std::int64_t B_size = req_sizes.B_size;

    alpha = alpha ? 1 : 0;

    auto A  = get_random_vector<float>(A_size);
    auto B  = get_random_vector<float>(B_size);
    auto CN = get_random_vector<float>(C_size);
    auto CJ = CN;

    auto jit_fn = loop_nest_code_generator<ISA>(args, fma, max_unrolled_fmas)
                      .get_shared();

    jit_fn.save_to_file("zi.asm");

    auto baseline_fn =
        loop_nest_baseline(order, sizes, C_formula, A_formula, B_formula,
                           C_strides, A_strides, B_strides, false);

    jit_fn(CJ.data(), A.data(), B.data(), alpha);
    baseline_fn(CN.data(), A.data(), B.data(), alpha);

    static int it = 0;
    std::cout << (it++) << std::endl;

    REQUIRE(max_abs_difference(CJ.data(), CJ.data() + C_size, CN.data()) <
            1e-5 * args.reduction_size());
}

} // namespace dabun::tests::baseline

TEST_CASE("ok?", "[wtfgh]")
{
    using namespace dabun;

    // return 0;

    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 333;
        int AcBr = 333;
        int BcCc = 133;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            // The first argument is the loop order in the form of
            // {dimension, stride}.  For now the outer dimension
            // has to divide the stride.  This is effectively the
            // same as Halide's split into outer and inner
            // variable, but can have arbitray number of splits.
            {{"ArCr", 11},
             {"AcBr", 11},
             {"ArCr", 5},
             {"AcBr", 5},
             {"ArCr", 2},
             {"AcBr", 2},

             {"BcCc", 16},
             {"AcBr", 1}, // inner loops, should handle
                          // differently later
             {"ArCr", 1},
             {"BcCc", 1}},
            // The second argument is a map of the dimension sizes
            {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
            // Vars of C (other variables are reduction variables)
            {"ArCr", "BcCc"},
            // Variables of A
            {"ArCr", "AcBr"},
            // Variables of B
            {"AcBr", "BcCc"},
            // C's strides for each variable.  Note that the
            // strides data is a superset of the previous argument
            // (variables of C).  I'm still deciding on the final
            // design, possibly allowing for null strides that
            // will just deduce them from the sizes, or some
            // special structs indicating the layout (ie
            // row-major, col-major).  In this case the vars have
            // to be ordered though... Many decisions to make...
            {{"ArCr", BcCc}, {"BcCc", 1}},
            // A's strides for each variable
            {{"ArCr", AcBr}, {"AcBr", 1}},
            // B's strides for each variable
            {{"AcBr", BcCc}, {"BcCc", 1}}, 2);
    }

    // 2D convolution NCHW example:
    // O(c_out, o_h, o_w) = I(c_i, o_h + k_h, ow + k_w) * K(c_o, c_i,
    // k_h, k_w)
    {
        int CIN  = 33;
        int COUT = 41;
        int OS   = 37;
        int KS   = 3;
        int IS   = OS + KS - 1;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
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
            32);
    }

    // outer product M(r, c) = A(r) * B(c)
    {
        int r = 133;
        int c = 133;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            {{"c", 1},  //
             {"r", 1}}, //
            {{"r", r}, {"c", c}},
            // Vars of C (other variables are reduction variables)
            {"r", "c"},
            // Variables of A
            {"r"},
            // Variables of B
            {"c"},
            // C's strides for each variable
            {{"r", c}, {"c", 1}},
            // A's strides for each variable
            {{"r", 1}},
            // B's strides for each variable
            {{"c", 1}});
    }

    // return 0;

    // Simple reduction of matrix columns using the FMA loop nest
    // The trick is to use a fake tensor "A" - that is a tensor with
    // a single element and 0 strides.
    {
        int ArCr = 1;
        int AcBr = 333;
        int BcCc = 333;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            {{"AcBr", 512},
             {"BcCc", (std::is_same_v<DABUN_ISA, avx2> ? 8 : 16) * 10},
             {"AcBr", 1},
             {"ArCr", 1},
             {"BcCc", 1}},
            // The second argument is a map of the dimension sizes
            {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
            // Vars of C (other variables are reduction variables)
            {"ArCr", "BcCc"},
            // Variables of A
            {"ArCr", "AcBr"},
            // Variables of B
            {"AcBr", "BcCc"},
            // C's strides for each variable.  Note that the
            // strides data is a superset of the previous argument
            // (variables of C).  I'm still deciding on the final
            // design, possibly allowing for null strides that
            // will just deduce them from the sizes, or some
            // special structs indicating the layout (ie
            // row-major, col-major).  In this case the vars have
            // to be ordered though... Many decisions to make...
            {{"ArCr", BcCc}, {"BcCc", 1}},
            // A's strides for each variable
            {{"ArCr", 0}, {"AcBr", 0}},
            // B's strides for each variable
            {{"AcBr", 1}, {"BcCc", AcBr}}, 512, 1);
    }

    // WOW this is actually pretty efficient!
    // Playing with weird schedules
    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 333;
        int AcBr = 333;
        int BcCc = 333;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            // The first argument is the loop order in the form of
            // {dimension, stride}.  For now the outer dimension
            // has to divide the stride.  This is effectively the
            // same as Halide's split into outer and inner
            // variable, but can have arbitray number of splits.
            {{"AcBr", 128},
             {"ArCr", std::is_same_v<DABUN_ISA, avx2> ? 12 : 28},
             {"BcCc", std::is_same_v<DABUN_ISA, avx2> ? 8 : 16},
             {"AcBr", 1},
             {"ArCr", 1},
             {"BcCc", 1}},
            // The second argument is a map of the dimension sizes
            {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
            // Vars of C (other variables are reduction variables)
            {"ArCr", "BcCc"},
            // Variables of A
            {"ArCr", "AcBr"},
            // Variables of B
            {"AcBr", "BcCc"},
            // C's strides for each variable.  Note that the
            // strides data is a superset of the previous argument
            // (variables of C).  I'm still deciding on the final
            // design, possibly allowing for null strides that
            // will just deduce them from the sizes, or some
            // special structs indicating the layout (ie
            // row-major, col-major).  In this case the vars have
            // to be ordered though... Many decisions to make...
            {{"ArCr", 1}, {"BcCc", ArCr}},
            // A's strides for each variable
            {{"ArCr", 1}, {"AcBr", ArCr}},
            // B's strides for each variable
            {{"AcBr", 1}, {"BcCc", AcBr}}, 512);
    }

    // return 0;

    // Playing with weird schedules
    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 333;
        int AcBr = 333;
        int BcCc = 333;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            // The first argument is the loop order in the form of
            // {dimension, stride}.  For now the outer dimension
            // has to divide the stride.  This is effectively the
            // same as Halide's split into outer and inner
            // variable, but can have arbitray number of splits.
            {{"ArCr", 123},
             {"AcBr", 123},
             {"ArCr", 17},
             {"AcBr", 17},
             {"ArCr", 7},
             {"AcBr", 7},
             {"ArCr", 3},
             {"AcBr", 3},
             {"ArCr", 2},
             {"AcBr", 2},

             {"AcBr", 1}, // inner loops, should handle
                          // differently later
             {"ArCr", 1},
             {"BcCc", 112}, // TODO DEBUG NOT INJECTING THE LOOP!!!!!
             {"BcCc", 1}},
            // The second argument is a map of the dimension sizes
            {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
            // Vars of C (other variables are reduction variables)
            {"ArCr", "BcCc"},
            // Variables of A
            {"ArCr", "AcBr"},
            // Variables of B
            {"AcBr", "BcCc"},
            // C's strides for each variable.  Note that the
            // strides data is a superset of the previous argument
            // (variables of C).  I'm still deciding on the final
            // design, possibly allowing for null strides that
            // will just deduce them from the sizes, or some
            // special structs indicating the layout (ie
            // row-major, col-major).  In this case the vars have
            // to be ordered though... Many decisions to make...
            {{"ArCr", BcCc}, {"BcCc", 1}},
            // A's strides for each variable
            {{"ArCr", AcBr}, {"AcBr", 1}},
            // B's strides for each variable
            {{"AcBr", BcCc}, {"BcCc", 1}}, 512);
    }

    // return 0;

    // (row-major)Matrix-(column)Vector product (requires horizontal
    // sum) C(r) = A(r, k) * B(k) if (0)
    {
        int ArCr = 256 + 3;
        int AcBr = 256 + 3;
        // int BcCc = 1;

        int k = AcBr;
        int r = ArCr;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            {{"r", 16}, //
             {"r", 1},  //
             {"k", 64},
             {"k", 1}}, //
            {{"k", k}, {"r", r}},
            // Vars of C (other variables are reduction variables)
            {"r"},
            // Variables of A
            {"r", "k"},
            // Variables of B
            {"k"},
            // C's strides for each variable
            {{"r", 1}},
            // A's strides for each variable
            {{"r", k}, {"k", 1}},
            // B's strides for each variable
            {{"k", 1}});
    }

    // return 0;

    // (row-major)Matrix-(col-major)Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 120 * 32;
        int AcBr = 256 + 3;
        int BcCc = 256 + 3;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            // The first argument is the loop order in the form of
            // {dimension, stride}.  For now the outer dimension
            // has to divide the stride.  This is effectively the
            // same as Halide's split into outer and inner
            // variable, but can have arbitray number of splits.
            {{"ArCr", 16}, // This and the next are for the
                           // register blocking of C - 30 vector
                           // registers of each holding 16 values
             {"BcCc", 16},
             {"ArCr", 1},
             {"BcCc", 1},
             {"AcBr", 1}},
            // The second argument is a map of the dimension sizes
            {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
            // Vars of C (other variables are reduction variables)
            {"ArCr", "BcCc"},
            // Variables of A
            {"ArCr", "AcBr"},
            // Variables of B
            {"AcBr", "BcCc"},
            // C's strides for each variable.  Note that the
            // strides data is a superset of the previous argument
            // (variables of C).  I'm still deciding on the final
            // design, possibly allowing for null strides that
            // will just deduce them from the sizes, or some
            // special structs indicating the layout (ie
            // row-major, col-major).  In this case the vars have
            // to be ordered though... Many decisions to make...
            {{"ArCr", BcCc}, {"BcCc", 1}},
            // A's strides for each variable
            {{"ArCr", AcBr}, {"AcBr", 1}},
            // B's strides for each variable
            {{"AcBr", 1}, {"BcCc", AcBr}});
    }

    // return 0;

    // return 0;

    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 1;
        int AcBr = 1;
        int BcCc = 256 + 251;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            // The first argument is the loop order in the form of
            // {dimension, stride}.  For now the outer dimension
            // has to divide the stride.  This is effectively the
            // same as Halide's split into outer and inner
            // variable, but can have arbitray number of splits.
            {{"AcBr", 1}, // inner loops, should handle
                          // differently later
             {"ArCr", 1},
             {"BcCc", 1}},
            // The second argument is a map of the dimension sizes
            {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
            // Vars of C (other variables are reduction variables)
            {"ArCr", "BcCc"},
            // Variables of A
            {"ArCr", "AcBr"},
            // Variables of B
            {"AcBr", "BcCc"},
            // C's strides for each variable.  Note that the
            // strides data is a superset of the previous argument
            // (variables of C).  I'm still deciding on the final
            // design, possibly allowing for null strides that
            // will just deduce them from the sizes, or some
            // special structs indicating the layout (ie
            // row-major, col-major).  In this case the vars have
            // to be ordered though... Many decisions to make...
            {{"ArCr", BcCc}, {"BcCc", 1}},
            // A's strides for each variable
            {{"ArCr", AcBr}, {"AcBr", 1}},
            // B's strides for each variable
            {{"AcBr", BcCc}, {"BcCc", 1}});
    }

    // return 0;

    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 120 * 4 + 3;
        int AcBr = 256 + 3;
        int BcCc = 259;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            // The first argument is the loop order in the form of
            // {dimension, stride}.  For now the outer dimension
            // has to divide the stride.  This is effectively the
            // same as Halide's split into outer and inner
            // variable, but can have arbitray number of splits.
            {{"AcBr", 132}, // To block B in L2 cache
             {"ArCr", 30},  // This and the next are for the
                            // register blocking of C - 30 vector
                            // registers of each holding 16 values
             {"BcCc", 16},
             {"AcBr", 4}, // broken up to allow for unrolling of 4
             {"AcBr", 1}, // inner loops, should handle
                          // differently later
             {"ArCr", 1},
             {"BcCc", 1}},
            // The second argument is a map of the dimension sizes
            {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
            // Vars of C (other variables are reduction variables)
            {"ArCr", "BcCc"},
            // Variables of A
            {"ArCr", "AcBr"},
            // Variables of B
            {"AcBr", "BcCc"},
            // C's strides for each variable.  Note that the
            // strides data is a superset of the previous argument
            // (variables of C).  I'm still deciding on the final
            // design, possibly allowing for null strides that
            // will just deduce them from the sizes, or some
            // special structs indicating the layout (ie
            // row-major, col-major).  In this case the vars have
            // to be ordered though... Many decisions to make...
            {{"ArCr", BcCc}, {"BcCc", 1}},
            // A's strides for each variable
            {{"ArCr", AcBr}, {"AcBr", 1}},
            // B's strides for each variable
            {{"AcBr", BcCc}, {"BcCc", 1}});
    }

    // return 0;

    // Single 3D convolution
    // C(x, y, z) = A(x + x_k, y + y_k, z + z_k) *
    //              B(x_k, y_k, z_k)
    // if (0)
    {
        int OX = 101;
        int OY = 101;
        int OZ = 16 * 12 + 3;
        int KX = 3;
        int KY = 3;
        int KZ = 3;
        int IX = OX + KX - 1;
        int IY = OY + KY - 1;
        int IZ = OZ + KZ - 1;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            // The first argument is the loop order in the form of
            // {dimension, stride}.  For now the outer dimension has
            // to divide the stride.  This is effectively the same as
            // Halide's split into outer and inner variable, but can
            // have arbitray number of splits.
            {{"OX", 1},  // To block B in L2 cache
             {"OY", 10}, // This and the next are for the register
                         // blocking of C - 30 vector registers of
                         // each holding 16 values
             {"OY", 1},
             {"OZ", 16},
             {"KX", 1}, // broken up to allow for unrolling of 4
             {"KY", 1}, // inner loops, should handle differently
                        // later
             {"KZ", 1},
             {"OZ", 1}},
            // The second argument is a map of the dimension sizes
            {{"OX", OX},
             {"OY", OY},
             {"OZ", OZ},
             {"IX", IX},
             {"IY", IY},
             {"IZ", IZ},
             {"KX", KX},
             {"KY", KY},
             {"KZ", KZ}},
            // Vars of C (other variables are reduction variables)
            {"OX", "OY", "OZ"},
            // Variables of A
            {"IX", "IY", "IZ"},
            // Variables of B
            {"KX", "KY", "KZ"},
            // C's strides for each variable.  Note that the strides
            // data is a superset of the previous argument (variables
            // of C).  I'm still deciding on the final design,
            // possibly allowing for null strides that will just
            // deduce them from the sizes, or some special structs
            // indicating the layout (ie row-major, col-major).  In
            // this case the vars have to be ordered though...
            // Many decisions to make...
            {{"OX", OY * OZ}, {"OY", OZ}, {"OZ", 1}},
            // A's strides for each variable
            {{"OX", IY * IZ},
             {"OY", IZ},
             {"OZ", 1},
             {"KX", IY * IZ},
             {"KY", IZ},
             {"KZ", 1}},
            // B's strides for each variable
            {{"KX", KY * KZ}, {"KY", KZ}, {"KZ", 1}});
    }

    // return 0;

    // (row)Vector-(row-major)Matrix product
    // C(c) = A(k) * B(k, c)
    // if (0)
    {
        // int ArCr = 1;
        int AcBr = 64 * 128;
        int BcCc = 16 + 7;

        int k = AcBr;
        int c = BcCc;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            {{"k", 64}, //
             {"k", 1},  //
             {"c", 1}}, //
            {{"k", k}, {"c", c}},
            // Vars of C (other variables are reduction variables)
            {"c"},
            // Variables of A
            {"k"},
            // Variables of B
            {"c", "k"},
            // C's strides for each variable
            {{"c", 1}},
            // A's strides for each variable
            {{"k", 1}},
            // B's strides for each variable
            {{"k", c}, {"c", 1}});
    }

    // return 0;

    // (row)Vector-(row-major)Matrix product
    // C(c) = A(k) * B(k, c)
    // if (0)
    {
        // int ArCr = 1;
        int AcBr = 64;
        int BcCc = 16 * 28 + 3;

        int k = AcBr;
        int c = BcCc;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            {{"k", 4},  //
             {"k", 1},  //
             {"c", 1}}, //
            {{"k", k}, {"c", c}},
            // Vars of C (other variables are reduction variables)
            {"c"},
            // Variables of A
            {"k"},
            // Variables of B
            {"c", "k"},
            // C's strides for each variable
            {{"c", 1}},
            // A's strides for each variable
            {{"k", 1}},
            // B's strides for each variable
            {{"k", c}, {"c", 1}});
    }

    // return 0;

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

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
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

    // 2D convolution NHWC example:
    // O(c_out, o_h, o_w) = I(c_i, o_h + k_h, ow + k_w) * K(c_o, c_i,
    // k_h, k_w) if (0)
    {
        int CIN  = 128;
        int COUT = 128 + 3;
        int OS   = 56 + 4;
        int KS   = 3;
        int IS   = OS + KS - 1;

        tests::baseline::loop_nest_against_slow_baseline<DABUN_ISA>(
            {{"c_out", 16}, //
             {"o_h", 1},
             {"o_w", 28},
             {"c_in", 16},
             {"c_in", 1},
             {"o_w", 1}, //
             //{"o_w", 1},    //
             {"k_h", 1},    //
             {"k_w", 1},    //
             {"c_out", 1}}, //
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
            {{"o_w", COUT}, {"c_out", 1}, {"o_h", COUT * OS}},
            // A's strides for each variable Note how we
            // provide strides for i/k_h and i/k_w, this is
            // because the access to A is based on output
            // and reduction variables
            {{"o_w", CIN},
             {"k_w", CIN},
             {"c_in", 1},
             {"o_h", IS * CIN},
             {"k_h", IS * CIN}},
            // B's strides for each variable
            {{"c_out", 1},
             {"c_in", COUT},
             {"k_w", COUT * CIN},
             {"k_h", COUT * CIN * KS}});
    }

    REQUIRE(1 == 1);
}
