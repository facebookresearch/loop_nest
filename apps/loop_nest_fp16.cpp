// Copyright 2004-present Facebook. All Rights Reserved.

#include "baselines.hpp"
#include "dabun/arithmetic_operation.hpp"
#include "dabun/check.hpp"
#include "dabun/elementwise_operation.hpp"
#include "dabun/loop_nest.hpp"
#include "dabun/measure.hpp"
#include "dabun/one_constant.hpp"
#include "dabun/random_vector.hpp"
#include "loop_nest_baseline.hpp"
#include "utility.hpp"

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

// WHEN ALL TENSORS VECTORIZED USE NEW SCHEDULER

int main()
{
    using namespace dabun;
    {
        std::cout << "Benchmark: 14" << std::endl;

        int ArCr = 1;
        int AcBr = 64 * 128;
        int BcCc = 16 + 7;

        int k = AcBr;
        int c = BcCc;

        auto gen_loop_nest = [&]() {
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"k", c}, {"c", 1}}, dabun::fma)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        fn.register_perf("fn8");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CJ = std::vector<fp16>(ArCr * BcCc);
        auto CN = CJ;

        baseline_MM(ArCr, AcBr, BcCc, 0, 1, c, 1, 0, 1, A.data(), B.data(),
                    CN.data(), 0);

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    return 0;


    // if (0)
    {
        int ArCr = 1;//123;
        int AcBr = 1; //123;
        int BcCc = 64;

        auto gen_loop_nest = [&]() {
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       {{"ArCr", 6},
                        {"BcCc", 32},
                        {"AcBr", 6},
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, dabun::fma, 372, nullptr,
                       {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto A_fp16 = aligned_vector_cast<fp16>(A);
        auto B_fp16 = aligned_vector_cast<fp16>(B);

        auto CN = get_random_vector<fp16>(ArCr * BcCc);
        auto CJ = aligned_vector_cast<fp16>(CN);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data(), 1);

        apply_relu(CN.data(), CN.data() + CN.size());

        fn(CJ.data(), A_fp16.data(), B_fp16.data(), 1);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference_verbose(CN.data(), CN.data() + ArCr * BcCc,
                                        CJ.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A_fp16.data(), B_fp16.data(), 0); }, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(
        //     fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
        //     1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
    }

    // return 0;

    {
        int ArCr = 8; // 126; // 123;
        int AcBr = 8; // 124;  // 126; // 123;
        int BcCc = 62;

        auto gen_loop_nest = [&]() {
            return dabun::loop_nest_code_generator<DABUN_ISA>(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       {{"ArCr", 8},
                        {"BcCc", 2},
                        {"AcBr", 32},
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
                       {{"AcBr", 1}, {"BcCc", ArCr}}, dabun::fma, 3 /* 372 */,
                       nullptr, {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        // auto compile_secs = measure_fastest(gen_loop_nest, 1);
        // std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = get_random_vector<float>(AcBr * ArCr);
        auto B = get_random_vector<float>(AcBr * BcCc);

        auto A_float = aligned_vector_cast<float>(A);
        auto B_float = aligned_vector_cast<float>(B);

        auto CN = get_random_vector<float>(ArCr * BcCc);
        auto CJ = aligned_vector_cast<float>(CN);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, 1, ArCr, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        apply_relu(CN.data(), CN.data() + CN.size());

        fn(CJ.data(), A_float.data(), B_float.data(), 1);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CN.data(), CN.data() + ArCr * BcCc,
                                        CJ.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A_float.data(), B_float.data(), 0); }, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(
        //     fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
        //     1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                       {{"AcBr", 512},
                        {"BcCc",
                         (std::is_same_v<DABUN_ISA, avx2> ? 8 : 16) * 10},
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
                       {{"AcBr", 1}, {"BcCc", AcBr}}, dabun::fma, 512, nullptr,
                       {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        fp16 A = 1.f;

        auto B  = get_random_vector<fp16>(AcBr * BcCc);
        auto CN = get_random_vector<fp16>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, 0, 0, 1, AcBr, BcCc, 1, &A, B.data(),
                    CN.data(), 1);

        fn(CJ.data(), &A, B.data(), 1);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), &A, B.data(), 0); }, 100);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    {
        std::cout << "Benchmark: 10 (has horizontal add)" << std::endl;

        int ArCr = 133;
        int AcBr = 26 + 3;
        int BcCc = 256 + 3;

        auto gen_loop_nest = [&]() {
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"AcBr", 1}, {"BcCc", AcBr}}, dabun::fma, 1024, nullptr,
                       {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn4");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = std::vector<fp16>(ArCr * BcCc);
        auto CJ = std::vector<fp16>(ArCr * BcCc);

        baseline_MM_row_col_major(ArCr, AcBr, BcCc, AcBr, AcBr, BcCc, A.data(),
                                  B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), dabun::skip_postop);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    {
        std::cout << "Benchmark: 10" << std::endl;

        int ArCr = 120 * 32;
        int AcBr = 256 + 3;
        int BcCc = 256 + 3;

        auto gen_loop_nest = [&]() {
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"AcBr", 1}, {"BcCc", AcBr}}, dabun::fma, 1024, nullptr,
                       {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn4");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = std::vector<fp16>(ArCr * BcCc);
        auto CJ = std::vector<fp16>(ArCr * BcCc);

        baseline_MM_row_col_major(ArCr, AcBr, BcCc, AcBr, AcBr, BcCc, A.data(),
                                  B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), dabun::skip_postop);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, dabun::fma, 512, nullptr,
                       {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = get_random_vector<fp16>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(
        //     fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
        //     1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
    }

    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: 8" << std::endl;

        int ArCr = 333;
        int AcBr = 333;
        int BcCc = 133;

        auto gen_loop_nest = [&]() {
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, dabun::fma, 2, nullptr,
                       {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn2");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = std::vector<fp16>(ArCr * BcCc);
        auto CJ = std::vector<fp16>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       {{"AcBr", 256},
                        {"ArCr", 6},
                        {"BcCc", 2 * 8},
                        {"AcBr", 16},
                        {"AcBr", 1},
                        {"ArCr", 1},
                        {"BcCc", 1}},

                       // {{"AcBr", 256},
                       //  {"ArCr", 4},
                       //  {"BcCc", 3 * 16},
                       //  {"AcBr", 16},
                       //  {"AcBr", 1},
                       //  {"ArCr", 1},
                       //  {"BcCc", 1}},

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, dabun::fma, 256)
                .get_unique();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(fp16*, fp16 const*, fp16 const*, int)>(
            std::move(fny));

        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = get_random_vector<fp16>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, ArCr, 1, A.data(),
                    B.data(), CN.data(), 0);

        // apply_relu(CN.data(), CN.data() + CN.size());

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << " --- " << std::endl;

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 100);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(
        //     fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
        //     1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
    }

    // WOW this is actually pretty efficient!
    // Playing with weird schedules
    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        std::cout << "Benchmark: Depthwise" << std::endl;

        // int ArCr = 16;
        // int AcBr = 16;
        // int BcCc = 16;

        int OHW = 8;
        int KHW = 3;
        int IHW = OHW + KHW - 1;
        int IOC = 32;

        auto gen_loop_nest = [&]() {
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       {{"IOC", 32},
                        {"OH", 2},
                        {"OW", 3},
                        {"OH", 1},
                        {"OW", 1},
                        {"KH", 1},
                        {"KW", 1},
                        {"IOC", 1}},

                       // The second argument is a map of the dimension sizes
                       {{"OH", OHW},
                        {"OW", OHW},
                        {"KH", KHW},
                        {"KW", KHW},
                        {"IOC", IOC}},
                       // Vars of C (other variables are reduction variables)
                       {"OH", "OW", "IOC"},
                       // Variables of A
                       {"IH", "IW", "IOC"},
                       // Variables of B
                       {"KH", "KW", "IOC"},
                       // C's strides for each variable.
                       {{"OW", IOC}, {"OH", IOC * OHW}, {"IOC", 1}},
                       // A's strides for each variable
                       {{"OW", IOC},
                        {"KW", IOC},
                        {"OH", IOC * IHW},
                        {"KH", IOC * IHW},
                        {"IOC", 1}},
                       // B's strides for each variable
                       {{"KW", IOC}, {"KH", IOC * KHW}, {"IOC", 1}}, dabun::fma,
                       128)
                .get_unique();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();

        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = get_random_vector<fp16>(IHW * IHW * IOC);
        auto B = get_random_vector<fp16>(KHW * KHW * IOC);

        auto CN = get_random_vector<fp16>(OHW * OHW * IOC);
        auto CJ = CN;

        baseline_CW_HWC(IOC, OHW, KHW, A.data(), B.data(), CN.data(), 0);

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + OHW * OHW * IOC,
                                        CN.data())
                  << " --- " << std::endl;

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 100);

        double gflops = 1.0 * OHW * OHW * KHW * KHW * IOC * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << std::endl;
    }

    // (row-major)Matrix-(col-major)Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)

    // return 0;

    {
        /*
         */
        /*
         */
        /*
        a:0: 1
        b:0: 8
        b:1: 16
        c:0: 2
        c:1: 4
        c:2: 16
        d:0: 2
        d:1: 4
        d:2: 16
        e:0: 2
        e:1: 64

        order:
        c:0: 1
        d:0: 1
        c:1: 1
        c:2: 1
        a:0: 1
        d:1: 1
        d:2: 1
        15: a:0 s 128
        15: d:0 s 64
        15: d:1 s 16
        15: d:2 s 1

        28: a:0 s 128
        28: c:0 s 64
        28: c:1 s 16
        28: c:2 s 1

        25: c:0 s 8192
        25: c:1 s 2048
        25: c:2 s 128
        25: d:0 s 64
        25: d:1 s 16
        25: d:2 s 1
        */
        auto gen_loop_nest = [&]() {
            return arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                       {{"c:0", 1},
                        {"d:0", 1},
                        {"c:1", 1},
                        {"c:2", 1},
                        {"a:0", 1},
                        {"d:1", 1},
                        {"d:2", 1}},
                       // The second argument is a map of the dimension sizes
                       {{"a:0", 1},
                        {"b:0", 8},
                        {"b:1", 16},
                        {"c:0", 2},
                        {"c:1", 4},
                        {"c:2", 16},
                        {"d:0", 2},
                        {"d:1", 4},
                        {"d:2", 16},
                        {"e:0", 2},
                        {"e:1", 64}},
                       // Vars of C (other variables are reduction variables)
                       {"a:0", "d:0", "d:1", "d:2"},
                       // Variables of A
                       {"a:0", "c:0", "c:1", "c:2"},
                       // Variables of B
                       {"c:0", "c:1", "c:2", "d:0", "d:1", "d:2"},

                       {{"a:0", 128}, {"d:0", 64}, {"d:1", 16}, {"d:2", 1}},
                       // A's strides for each variable
                       {{"a:0", 128}, {"c:0", 64}, {"c:1", 16}, {"c:2", 1}},
                       // B's strides for each variable
                       {{"c:0", 8192},
                        {"c:1", 2048},
                        {"c:2", 128},
                        {"d:0", 64},
                        {"d:1", 16},
                        {"d:2", 1}},
                       dabun::fma, 320, nullptr, {},
                       dabun::elementwise_relu<DABUN_ISA>)
                .get_unique();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();

        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto M = 1;
        auto N = 128;
        auto K = 128;
        auto A = get_random_vector<fp16>(M * K);
        for (auto i = 0; i < M * K; ++i)
        {
            A.data()[i] = 1;
        }
        auto B = get_random_vector<fp16>(K * N);
        for (auto i = 0; i < N * K; ++i)
        {
            B.data()[i] = 1;
        }

        auto CN = get_random_vector<fp16>(M * N);
        auto CJ = get_random_vector<fp16>(M * N);

        baseline_MM(M, K, N, K, 1, N, 1, N, 1, A.data(), B.data(), CN.data(),
                    0);

        apply_relu(CN.data(), CN.data() + CN.size());

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + M * N, CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 100);

        double gflops = 1.0 * K * M * N * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(fn, K * M, K * N, M * N,
        //                                     1.0 * K * M * N * 2, 10, 10);
    }

    /// return 0;

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                        {"k_h", COUT * CIN * KS}},
                       dabun::fma)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn10");

        auto A  = get_random_vector<fp16>(CIN * IS * IS);
        auto B  = get_random_vector<fp16>(COUT * CIN * KS * KS);
        auto CN = std::vector<fp16>(COUT * OS * OS);
        auto CJ = std::vector<fp16>(COUT * OS * OS);

        baseline_Conv(COUT, CIN, OS, OS, KS, KS, A.data(), B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + COUT * OS * OS,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1);

        double gflops = 2.0 * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // 2D convolution example:
    // O(c_out, o_h, o_w) = I(c_i, o_h + k_h, ow + k_w) * K(c_o, c_i,
    // k_h, k_w) if (0)
    {
        std::cout << "Benchmark Padded Conv: 16" << std::endl;

        int CIN  = 128;
        int COUT = 128 + 3;
        int OS   = 56 + 4;
        int KS   = 3;

        int PS = 1;
        int IS = OS + KS - 1 - 2 * PS;

        strong_assert(IS == OS);

        std::set<std::string> C_formula = {"c_out", "o_w", "o_h"};
        std::set<std::string> A_formula = {"c_in", "i_w", "i_h"};
        std::set<std::string> B_formula = {"c_in", "c_out", "k_w", "k_h"};

        std::map<std::string, int> C_strides = {
            {"o_w", COUT}, {"c_out", 1}, {"o_h", COUT * OS}};
        std::map<std::string, int> A_strides = {{"o_w", CIN},
                                                {"k_w", CIN},
                                                {"c_in", 1},
                                                {"o_h", IS * CIN},
                                                {"k_h", IS * CIN}};
        std::map<std::string, int> B_strides = {{"c_out", 1},
                                                {"c_in", COUT},
                                                {"k_w", COUT * CIN},
                                                {"k_h", COUT * CIN * KS}};

        auto fn_c = dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                         {"o_w", OS - 2 * PS},
                         {"k_w", KS},
                         {"c_in", CIN},
                         {"o_h", OS - 2 * PS},
                         {"k_h", KS}},

                        C_formula, A_formula, B_formula, C_strides, A_strides,
                        B_strides, dabun::fma)
                        .get_shared();

        int in_c_off  = 0;
        int out_c_off = (PS * OS + PS) * COUT;
        int ker_c_off = 0;

        // Bottom left
        auto fn_corners =
            dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                 {"o_w", PS},
                 {"k_w", KS - PS},
                 {"c_in", CIN},
                 {"o_h", PS},
                 {"k_h", KS - PS}},

                C_formula, A_formula, B_formula, C_strides, A_strides,
                B_strides, dabun::fma)
                .get_shared();

        int in_bl_off  = 0;
        int out_bl_off = 0;
        int ker_bl_off = (PS * KS + PS) * CIN * COUT;

        int in_br_off  = (OS - 2 * PS) * CIN;
        int out_br_off = (OS - PS) * COUT;
        int ker_br_off = (PS * KS) * CIN * COUT;

        int in_tl_off  = (OS - 2 * PS) * OS * CIN;
        int out_tl_off = (OS - PS) * OS * COUT;
        int ker_tl_off = (PS)*CIN * COUT;

        int in_tr_off  = in_tl_off + in_br_off;
        int out_tr_off = out_tl_off + out_br_off;
        int ker_tr_off = 0;

        // Bottom-Top
        auto fn_bt =
            dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                {{"c_out", 16}, //
                 {"o_h", 1},
                 {"o_w", 28},
                 {"c_in", 16},
                 {"c_in", 1},
                 {"o_w", 1}, //
                 //{"o_w", 1},    //
                 {"k_h", 1}, //
                 {"k_w", 1}, //
                 {"c_out",
                  1}}, //
                       // The second argument is a map of the dimension sizes
                {{"c_out", COUT},
                 {"o_w", OS - 2 * PS},
                 {"k_w", KS},
                 {"c_in", CIN},
                 {"o_h", PS},
                 {"k_h", KS - PS}},

                C_formula, A_formula, B_formula, C_strides, A_strides,
                B_strides, dabun::fma)
                .get_shared();

        int in_b_off  = 0;
        int out_b_off = (PS)*COUT;
        int ker_b_off = (PS * KS) * CIN * COUT;

        int in_t_off  = (OS - 2 * PS) * OS * CIN;
        int out_t_off = ((OS - PS) * OS + PS) * COUT;
        int ker_t_off = 0;

        // Left-Right
        auto fn_lr =
            dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                {{"c_out", 16}, //
                 {"o_h", 1},
                 {"o_w", 28},
                 {"c_in", 16},
                 {"c_in", 1},
                 {"o_w", 1}, //
                 //{"o_w", 1},    //
                 {"k_h", 1}, //
                 {"k_w", 1}, //
                 {"c_out",
                  1}}, //
                       // The second argument is a map of the dimension sizes
                {{"c_out", COUT},
                 {"o_w", PS},
                 {"k_w", KS - PS},
                 {"c_in", CIN},
                 {"o_h", OS - 2 * PS},
                 {"k_h", KS}},

                C_formula, A_formula, B_formula, C_strides, A_strides,
                B_strides, dabun::fma)
                .get_shared();

        int in_l_off  = 0;
        int out_l_off = (PS * OS) * COUT;
        int ker_l_off = (PS)*CIN * COUT;

        int in_r_off  = (OS - 2 * PS) * CIN;
        int out_r_off = (PS * OS + (OS - PS)) * COUT;
        int ker_r_off = 0;

        fn_c.save_to_file("zi.asm");
        // fn.register_perf("fn10");

        auto A  = get_random_vector<fp16>(CIN * OS * OS);
        auto B  = get_random_vector<fp16>(COUT * CIN * KS * KS);
        auto CN = std::vector<fp16>(COUT * OS * OS);
        auto CJ = std::vector<fp16>(COUT * OS * OS);

        baseline_padded_Conv(COUT, CIN, OS, OS, KS, KS, PS, PS, A.data(),
                             B.data(), CN.data());

        auto do_it = [&]() {
            fn_corners(CJ.data() + out_bl_off, A.data() + in_bl_off,
                       B.data() + ker_bl_off, 0);

            fn_bt(CJ.data() + out_b_off, A.data() + in_b_off,
                  B.data() + ker_b_off, 0);

            fn_corners(CJ.data() + out_br_off, A.data() + in_br_off,
                       B.data() + ker_br_off, 0);

            fn_lr(CJ.data() + out_l_off, A.data() + in_l_off,
                  B.data() + ker_l_off, 0);

            fn_c(CJ.data() + out_c_off, A.data() + in_c_off,
                 B.data() + ker_c_off, 0);

            fn_lr(CJ.data() + out_r_off, A.data() + in_r_off,
                  B.data() + ker_r_off, 0);

            fn_corners(CJ.data() + out_tl_off, A.data() + in_tl_off,
                       B.data() + ker_tl_off, 0);

            fn_bt(CJ.data() + out_t_off, A.data() + in_t_off,
                  B.data() + ker_t_off, 0);

            fn_corners(CJ.data() + out_tr_off, A.data() + in_tr_off,
                       B.data() + ker_tr_off, 0);
        };

        // fn_c(CJ.data(), A.data(), B.data(), 0);

        do_it();

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + COUT * OS * OS,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest([&]() { do_it(); }, 10);

        double gflops = 2.0 * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

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

        int ArCr = 32;
        int AcBr = 512;
        int BcCc = 512;

        auto gen_loop_nest = [&]() {
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       {{"AcBr", 64},
                        {"ArCr", 4},
                        {"BcCc", 4 * 16},
                        {"AcBr", 16},
                        {"AcBr", 1},
                        {"ArCr", 1},
                        {"BcCc", 1}},

                       // {{"AcBr", 256},
                       //  {"ArCr", 4},
                       //  {"BcCc", 3 * 16},
                       //  {"AcBr", 16},
                       //  {"AcBr", 1},
                       //  {"ArCr", 1},
                       //  {"BcCc", 1}},

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, dabun::fma, 1024)
                .get_unique();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();

        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = get_random_vector<fp16>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);

        // apply_relu(CN.data(), CN.data() + CN.size());

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 100);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(
        //     fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
        //     1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
    }

    // return 0;

    // return 0;

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                       {{"g_out", 1}, //
                        {"o_h", 5},
                        {"o_w", 5},
                        {"g_in", 1},
                        {"c_in", 1},
                        {"k_h", 1}, //
                        {"k_w", 1}, //
                        {"o_h", 1},
                        {"o_w", 1}, //
                        //{"o_w", 1},    //
                        {"c_out", 1}}, //
                       // The second argument is a map of the dimension sizes
                       {{"g_out", GOUT},
                        {"c_out", COUT},
                        {"g_in", GIN},
                        {"c_in", CIN},
                        {"o_w", OS},
                        {"o_h", OS},
                        {"k_w", KS},
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
                       dabun::fma, 1024)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn9");

        auto A  = get_random_vector<fp16>(GIN * CIN * IS * IS);
        auto B  = get_random_vector<fp16>(GOUT * GIN * COUT * CIN * KS * KS);
        auto CN = std::vector<fp16>(GOUT * COUT * OS * OS);
        auto CJ = std::vector<fp16>(GOUT * COUT * OS * OS);

        baseline_Conv_NCHW8c(GOUT, COUT, GIN, CIN, OS, OS, KS, KS, A.data(),
                             B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);

        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + COUT * OS * OS,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 100);

        double gflops =
            2.0 * GIN * GOUT * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "gflops: " << gflops << "\n";

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
                       {{"AcBr", 512},
                        {"BcCc",
                         (std::is_same_v<DABUN_ISA, avx2> ? 8 : 16) * 10},
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
                       {{"AcBr", 1}, {"BcCc", AcBr}}, dabun::fma, 512, nullptr,
                       {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        fp16 A = 1.f;

        auto B  = get_random_vector<fp16>(AcBr * BcCc);
        auto CN = get_random_vector<fp16>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, 0, 0, 1, AcBr, BcCc, 1, &A, B.data(),
                    CN.data(), 1);

        fn(CJ.data(), &A, B.data(), 1);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), &A, B.data(), 0); }, 100);

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"AcBr", 1}, {"BcCc", AcBr}}, dabun::fma, 512, nullptr,
                       {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = get_random_vector<fp16>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, 1, ArCr, 1, AcBr, 1, ArCr, A.data(),
                    B.data(), CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(
        //     fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
        //     1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"k", 2}}, dabun::fma, 1024, nullptr, {},
                       dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn3");

        auto A = get_random_vector<fp16>(AcBr * ArCr * 2);
        auto B = get_random_vector<fp16>(AcBr * BcCc * 2);

        auto CN = std::vector<fp16>(ArCr * BcCc);
        auto CJ = std::vector<fp16>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, k * 2, 2, 2, 2, 1, 1, A.data(), B.data(),
                    CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 0);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"k", 1}}, dabun::fma)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn3");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = std::vector<fp16>(ArCr * BcCc);
        auto CJ = std::vector<fp16>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 100);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, dabun::fma)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn5");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = std::vector<fp16>(ArCr * BcCc + 16);
        auto CJ = std::vector<fp16>(ArCr * BcCc + 16);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1);

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, dabun::fma)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn6");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CN = std::vector<fp16>(ArCr * BcCc);
        auto CJ = std::vector<fp16>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1);

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"KX", KY * KZ}, {"KY", KZ}, {"KZ", 1}}, dabun::fma,
                       1024, nullptr, {}, dabun::elementwise_relu<DABUN_ISA>)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        fn.register_perf("fn7");

        auto A = get_random_vector<fp16>(IX * IY * IZ);
        auto B = get_random_vector<fp16>(KX * KY * KZ);

        auto CN = std::vector<fp16>(OX * OY * OZ);
        auto CJ = std::vector<fp16>(OX * OY * OZ);

        baseline_3DConv(OX, OY, OZ, KX, KY, KZ, A.data(), B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);
        apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + OX * OY * OZ,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1);

        double gflops = 1.0 * OX * OY * OZ * KX * KY * KZ * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(
        //     fn, IX * IY * IZ, KX * KY * KZ, OX * OY * OZ,
        //     1.0 * OX * OY * OZ * KX * KY * KZ * 2, 10, 10);
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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"k", c}, {"c", 1}}, dabun::fma)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        fn.register_perf("fn8");

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CJ = std::vector<fp16>(ArCr * BcCc);
        auto CN = CJ;

        baseline_MM(ArCr, AcBr, BcCc, 0, 1, c, 1, 0, 1, A.data(), B.data(),
                    CN.data(), 0);

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10);

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                       {{"k", c}, {"c", 1}}, dabun::fma)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();

        auto A = get_random_vector<fp16>(AcBr * ArCr);
        auto B = get_random_vector<fp16>(AcBr * BcCc);

        auto CJ = std::vector<fp16>(ArCr * BcCc);
        auto CN = CJ;

        baseline_MM(ArCr, AcBr, BcCc, 0, 1, c, 1, 0, 1, A.data(), B.data(),
                    CN.data(), 0);

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10);

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
            return dabun::arm::loop_nest_fp16_code_generator<DABUN_ISA>(
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
                        {"k_h", COUT * CIN * KS}},
                       dabun::fma)
                .get_shared();
        };

        auto compile_secs = measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        fn.save_to_file("zi.asm");
        // fn.register_perf("fn10");

        auto A  = get_random_vector<fp16>(CIN * IS * IS);
        auto B  = get_random_vector<fp16>(COUT * CIN * KS * KS);
        auto CN = std::vector<fp16>(COUT * OS * OS);
        auto CJ = std::vector<fp16>(COUT * OS * OS);

        baseline_Conv(COUT, CIN, OS, OS, KS, KS, A.data(), B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data(), 0);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + COUT * OS * OS,
                                        CN.data())
                  << "\n";

        auto secs =
            measure_fastest([&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 1);

        double gflops = 2.0 * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }
}
