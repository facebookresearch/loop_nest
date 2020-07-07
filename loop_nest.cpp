#include "baselines.h"
#include "loop_nest.h"
#include "loop_nest_baseline.h"
#include "one_constant.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024)
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

        baseline_MM(ArCr, AcBr, BcCc, 1, ArCr, 1, AcBr, 1, ArCr, A.data(),
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

        bench_implementation_fmas_per_cycle(
            fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
            1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
    }

    // return 0;

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
                        {"c_out", 1}})
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

    // return 0;

    // Simple reduction of matrix columns using the FMA loop nest
    // The trick is to use a fake tensor "A" - that is a tensor with
    // a single element and 0 strides.
    {
        std::cout << "Benchmark: 3" << std::endl;

        int ArCr = 1;
        int AcBr = 333;
        int BcCc = 333;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
                       {{"AcBr", 512},
                        {"BcCc", (std::is_same_v<CT_ISA, avx2> ? 8 : 16) * 10},
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 512)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();

        auto baseline_fn = facebook::sysml::aot::loop_nest_slow_baseline(
            {{"AcBr", 512},
             {"BcCc", (std::is_same_v<CT_ISA, avx2> ? 8 : 16) * 10},
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
            {{"AcBr", BcCc}, {"BcCc", 1}});

        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        // float A = 1.f;

        auto B  = getRandomVector<float>(AcBr * BcCc);
        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        // baseline_MM(ArCr, AcBr, BcCc, 0, 0, BcCc, 1, BcCc, 1, &A, B.data(),
        //             CN.data(), 1);

        using facebook::sysml::aot::one_constant;

        fn(CJ.data(), one_constant<float>, B.data(), 1);
        baseline_fn(CN.data(), one_constant<float>, B.data(), 1);

        // apply_relu(CN.data(), CN.data() + CN.size());
        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), one_constant<float>, B.data(), 0); }, 10,
            100);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // Simple reduction of matrix columns using the FMA loop nest
    // The trick is to use a fake tensor "A" - that is a tensor with
    // a single element and 0 strides.
    {
        std::cout << "Benchmark: 4" << std::endl;

        int ArCr = 1;
        int AcBr = 333;
        int BcCc = 333;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
                       {{"AcBr", 512},
                        {"BcCc", (std::is_same_v<CT_ISA, avx2> ? 8 : 16) * 10},
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
                       {{"AcBr", 1}, {"BcCc", AcBr}}, 512, nullptr, {},
                       facebook::sysml::aot::elementwise_relu<CT_ISA>)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        float A = 1.f;

        auto B  = getRandomVector<float>(AcBr * BcCc);
        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, 0, 0, 1, AcBr, BcCc, 1, &A, B.data(),
                    CN.data(), 1);

        fn(CJ.data(), &A, B.data(), 1);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), &A, B.data(), 0); }, 10, 100);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // WOW this is actually pretty efficient!
    // Playing with weird schedules
    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 5" << std::endl;

        int ArCr = 333;
        int AcBr = 333;
        int BcCc = 333;

        // int ArCr = 333;
        // int AcBr = 333;
        // int BcCc = 333;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       {{"AcBr", 128},
                        {"ArCr", std::is_same_v<CT_ISA, avx2> ? 12 : 28},
                        {"BcCc", std::is_same_v<CT_ISA, avx2> ? 8 : 16},
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
                       {{"AcBr", 1}, {"BcCc", AcBr}}, 512, nullptr, {},
                       facebook::sysml::aot::elementwise_relu<CT_ISA>)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, 1, ArCr, 1, AcBr, 1, ArCr, A.data(),
                    B.data(), CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        bench_implementation_fmas_per_cycle(
            fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
            1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
    }

    // return 0;

    // (row-major)Matrix-(column)Vector product (requires horizontal
    // sum) C(r) = A(r, k) * B(k) if (0)
    {
        std::cout << "Benchmark: 6" << std::endl;

        int ArCr = 333;
        int AcBr = 222;
        int BcCc = 1;

        int k = AcBr;
        int r = ArCr;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"r", k * 2}, {"k", 2}},
                       // B's strides for each variable
                       {{"k", 2}}, 1024, nullptr, {},
                       facebook::sysml::aot::elementwise_relu<CT_ISA>)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn3");

        auto A = getRandomVector<float>(AcBr * ArCr * 2);
        auto B = getRandomVector<float>(AcBr * BcCc * 2);

        auto CN = std::vector<float>(ArCr * BcCc);
        auto CJ = std::vector<float>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, k * 2, 2, 2, 2, 1, 1, A.data(), B.data(),
                    CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 0);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // Playing with weird schedules
    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 7" << std::endl;

        int ArCr = 333;
        int AcBr = 333;
        int BcCc = 333;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 512, nullptr, {},
                       facebook::sysml::aot::elementwise_relu<CT_ISA>)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        bench_implementation_fmas_per_cycle(
            fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
            1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
    }

    // return 0;

    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 8" << std::endl;

        int ArCr = 333;
        int AcBr = 333;
        int BcCc = 133;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 2, nullptr, {},
                       facebook::sysml::aot::elementwise_relu<CT_ISA>)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn2");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = std::vector<float>(ArCr * BcCc);
        auto CJ = std::vector<float>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // (row-major)Matrix-(column)Vector product (requires horizontal
    // sum) C(r) = A(r, k) * B(k) if (0)
    {
        std::cout << "Benchmark: 9" << std::endl;

        int ArCr = 256 + 3;
        int AcBr = 256 + 3;
        int BcCc = 1;

        int k = AcBr;
        int r = ArCr;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"k", 1}})
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn3");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = std::vector<float>(ArCr * BcCc);
        auto CJ = std::vector<float>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 100);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // (row-major)Matrix-(col-major)Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 10" << std::endl;

        int ArCr = 120 * 32;
        int AcBr = 256 + 3;
        int BcCc = 256 + 3;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"AcBr", 1}, {"BcCc", AcBr}}, 1024, nullptr, {},
                       facebook::sysml::aot::elementwise_relu<CT_ISA>)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn4");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = std::vector<float>(ArCr * BcCc);
        auto CJ = std::vector<float>(ArCr * BcCc);

        baseline_MM_row_col_major(ArCr, AcBr, BcCc, AcBr, AcBr, BcCc, A.data(),
                                  B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1, 1);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // return 0;

    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 11" << std::endl;

        int ArCr = 1;
        int AcBr = 1;
        int BcCc = 256 + 251;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
                       // The first argument is the loop order in the form
                       // of {dimension, stride}.  For now the outer
                       // dimension has to divide the stride.  This is
                       // effectively the same as Halide's split into outer
                       // and inner variable, but can have arbitray number
                       // of splits.
                       {{"AcBr", 1}, // inner loops, should handle
                                     // differently later
                        {"ArCr", 1},
                        {"BcCc", 1}},
                       // The second argument is a map of the dimension
                       // sizes
                       {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
                       // Vars of C (other variables are reduction
                       // variables)
                       {"ArCr", "BcCc"},
                       // Variables of A
                       {"ArCr", "AcBr"},
                       // Variables of B
                       {"AcBr", "BcCc"},
                       // C's strides for each variable.  Note that the
                       // strides data is a superset of the previous
                       // argument (variables of C).  I'm still deciding on
                       // the final design, possibly allowing for null
                       // strides that will just deduce them from the sizes,
                       // or some special structs indicating the layout (ie
                       // row-major, col-major).  In this case the vars have
                       // to be ordered though... Many decisions to make...
                       {{"ArCr", BcCc}, {"BcCc", 1}},
                       // A's strides for each variable
                       {{"ArCr", AcBr}, {"AcBr", 1}},
                       // B's strides for each variable
                       {{"AcBr", BcCc}, {"BcCc", 1}})
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn5");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = std::vector<float>(ArCr * BcCc + 16);
        auto CJ = std::vector<float>(ArCr * BcCc + 16);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1, 1);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 12" << std::endl;

        int ArCr = 120 * 4 + 3;
        int AcBr = 256 + 3;
        int BcCc = 259;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"AcBr", BcCc}, {"BcCc", 1}})
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn6");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = std::vector<float>(ArCr * BcCc);
        auto CJ = std::vector<float>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1, 1);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // Single 3D convolution
    // C(x, y, z) = A(x + x_k, y + y_k, z + z_k) *
    //              B(x_k, y_k, z_k)
    // if (0)
    {
        std::cout << "Benchmark: 13" << std::endl;

        int OX = 101;
        int OY = 101;
        int OZ = 16 * 12 + 3;
        int KX = 3;
        int KY = 3;
        int KZ = 3;
        int IX = OX + KX - 1;
        int IY = OY + KY - 1;
        int IZ = OZ + KZ - 1;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"KX", KY * KZ}, {"KY", KZ}, {"KZ", 1}}, 1024, nullptr,
                       {}, facebook::sysml::aot::elementwise_relu<CT_ISA>)
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        fn.register_perf("fn7");

        auto A = getRandomVector<float>(IX * IY * IZ);
        auto B = getRandomVector<float>(KX * KY * KZ);

        auto CN = std::vector<float>(OX * OY * OZ);
        auto CJ = std::vector<float>(OX * OY * OZ);

        baseline_3DConv(OX, OY, OZ, KX, KY, KZ, A.data(), B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + OX * OY * OZ, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1, 1);

        double gflops = 1.0 * OX * OY * OZ * KX * KY * KZ * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        bench_implementation_fmas_per_cycle(
            fn, IX * IY * IZ, KX * KY * KZ, OX * OY * OZ,
            1.0 * OX * OY * OZ * KX * KY * KZ * 2, 10, 10);
    }

    // return 0;

    // (row)Vector-(row-major)Matrix product
    // C(c) = A(k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 14" << std::endl;

        int ArCr = 1;
        int AcBr = 64 * 128;
        int BcCc = 16 + 7;

        int k = AcBr;
        int c = BcCc;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"k", c}, {"c", 1}})
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        fn.register_perf("fn8");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CJ = std::vector<float>(ArCr * BcCc);
        auto CN = CJ;

        baseline_MM(ArCr, AcBr, BcCc, 0, 1, c, 1, 0, 1, A.data(), B.data(),
                    CN.data(), 0);

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // (row)Vector-(row-major)Matrix product
    // C(c) = A(k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 15" << std::endl;

        int ArCr = 1;
        int AcBr = 64;
        int BcCc = 16 * 28 + 3;

        int k = AcBr;
        int c = BcCc;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                       {{"k", c}, {"c", 1}})
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CJ = std::vector<float>(ArCr * BcCc);
        auto CN = CJ;

        baseline_MM(ArCr, AcBr, BcCc, 0, 1, c, 1, 0, 1, A.data(), B.data(),
                    CN.data(), 0);

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // 2D convolution example:
    // O(c_out, o_h, o_w) = I(c_i, o_h + k_h, ow + k_w) * K(c_o, c_i,
    // k_h, k_w) if (0)
    {
        std::cout << "Benchmark: 16" << std::endl;

        int CIN  = 128;
        int COUT = 128 + 3;
        int OS   = 56 + 4;
        int KS   = 3;
        int IS   = OS + KS - 1;

        auto gen_loop_nest = [&]() {
            return facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                        {"k_h", KS}},
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
                        {"k_h", COUT * CIN * KS}})
                .get_shared();
        };

        auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn10");

        auto A  = getRandomVector<float>(CIN * IS * IS);
        auto B  = getRandomVector<float>(COUT * CIN * KS * KS);
        auto CN = std::vector<float>(COUT * OS * OS);
        auto CJ = std::vector<float>(COUT * OS * OS);

        baseline_Conv(COUT, CIN, OS, OS, KS, KS, A.data(), B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + COUT * OS * OS,
                                CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1, 1);

        double gflops = 2.0 * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }
}
