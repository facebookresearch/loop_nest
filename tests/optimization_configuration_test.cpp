#include "baselines.h"
#include "configuration.h"
#include "loop_nest.h"
#include "loop_nest_baseline.h"
#include "one_constant.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifndef CT_ISA
#define CT_ISA avx2
#endif

int main()
{
    using facebook::sysml::aot::aot_fn_cast;
    using facebook::sysml::aot::avx2;
    using facebook::sysml::aot::avx2_plus;
    using facebook::sysml::aot::avx512;

    // WOW this is actually pretty efficient!
    // Playing with weird schedules
    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 1" << std::endl;

        // int ArCr = 16;
        // int AcBr = 16;
        // int BcCc = 16;

        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 256;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       {{"AcBr", 256},
                        {"ArCr", 3},
                        {"BcCc", 16},
                        {"AcBr", 1},
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
                       {{"ArCr", AcBr}, {"AcBr", 1}},
                       // B's strides for each variable
                       {{"AcBr", BcCc}, {"BcCc", 1}}, facebook::sysml::aot::fma,
                       1024, nullptr, {}, nullptr, {},
                       facebook::sysml::aot::all_optims)
                .get_unique();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int)>(
            std::move(fny));

        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, ArCr, 1, A.data(),
                    B.data(), CN.data(), 0);

        // apply_relu(CN.data(), CN.data() + CN.size());

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 1000);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    {
        std::cout << "Benchmark: 1" << std::endl;

        // int ArCr = 16;
        // int AcBr = 16;
        // int BcCc = 16;

        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 256;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       {{"AcBr", 256},
                        {"ArCr", 3},
                        {"BcCc", 16},
                        {"AcBr", 1},
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
                       {{"ArCr", AcBr}, {"AcBr", 1}},
                       // B's strides for each variable
                       {{"AcBr", BcCc}, {"BcCc", 1}}, facebook::sysml::aot::fma,
                       1024, nullptr, {}, nullptr, {},
                       facebook::sysml::aot::no_optims)
                .get_unique();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int)>(
            std::move(fny));

        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, ArCr, 1, A.data(),
                    B.data(), CN.data(), 0);

        // apply_relu(CN.data(), CN.data() + CN.size());

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 1000);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // // // return 0;

    // 2D convolution on NCHW16c layout example:
    // O(g_out, c_out, o_h, o_w) = I(g_in, c_in, o_h + k_h, ow + k_w) *
    //                             K(g_in, g_out, c_in, c_out, k_h, k_w)
    // if (0)
    {
        std::cout << "Benchmark: 2" << std::endl;

        int GIN  = 128 / 16;
        int CIN  = 16;
        int GOUT = 128 / 16;
        int COUT = 16;
        int OS   = 56;
        int KS   = 3;
        int IS   = OS + KS - 1;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                        {"k_h", KS}},
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
                        {"c_out", 1}},
                       facebook::sysml::aot::fma, 1024, nullptr, {}, nullptr,
                       {}, facebook::sysml::aot::all_optims)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn9");

        auto A  = getRandomVector<float>(GIN * CIN * IS * IS);
        auto B  = getRandomVector<float>(GOUT * GIN * COUT * CIN * KS * KS);
        auto CN = std::vector<float>(GOUT * COUT * OS * OS);
        auto CJ = std::vector<float>(GOUT * COUT * OS * OS);

        baseline_Conv_NCHW8c(GOUT, COUT, GIN, CIN, OS, OS, KS, KS, A.data(),
                             B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);

        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + COUT * OS * OS,
                                CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1, 100);

        double gflops =
            2.0 * GIN * GOUT * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "gflops: " << gflops << "\n";

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // 2D convolution on NCHW16c layout example:
    // O(g_out, c_out, o_h, o_w) = I(g_in, c_in, o_h + k_h, ow + k_w) *
    //                             K(g_in, g_out, c_in, c_out, k_h, k_w)
    // if (0)
    {
        std::cout << "Benchmark: 2" << std::endl;

        int GIN  = 128 / 16;
        int CIN  = 16;
        int GOUT = 128 / 16;
        int COUT = 16;
        int OS   = 56;
        int KS   = 3;
        int IS   = OS + KS - 1;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                        {"k_h", KS}},
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
                        {"c_out", 1}},
                       facebook::sysml::aot::fma, 1024, nullptr, {}, nullptr,
                       {}, facebook::sysml::aot::no_optims)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn9");

        auto A  = getRandomVector<float>(GIN * CIN * IS * IS);
        auto B  = getRandomVector<float>(GOUT * GIN * COUT * CIN * KS * KS);
        auto CN = std::vector<float>(GOUT * COUT * OS * OS);
        auto CJ = std::vector<float>(GOUT * COUT * OS * OS);

        baseline_Conv_NCHW8c(GOUT, COUT, GIN, CIN, OS, OS, KS, KS, A.data(),
                             B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);

        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + COUT * OS * OS,
                                CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1, 100);

        double gflops =
            2.0 * GIN * GOUT * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "gflops: " << gflops << "\n";

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }
}
