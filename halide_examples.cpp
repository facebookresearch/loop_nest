// Copyright 2004-present Facebook. All Rights Reserved.

/*
Rewriting loop_nest.cpp examples into Halide for purposes of
benchmarking
*/

#include <algorithm>
#include <iostream>
#include <type_traits>

#include "Halide.h"
#include "baselines.h"
#include "isa.h"
#include "utils.h"

using facebook::sysml::aot::avx2;

#ifndef CT_ISA
#define CT_ISA avx2
#endif

void check_equivalence(Halide::Buffer<float> result,
                       float*                baseline_result_start)
{
    std::int64_t num_elements = result.size_in_bytes() / 4;
    float        diff = maxAbsDiff(result.data(), result.data() + num_elements,
                            baseline_result_start);
    std::cout << "diff: " << diff << std::endl;
    assert(diff <= 1e-4);
}

template <class Fn>
void benchmarking_stats(Fn&& fn, std::int64_t flops, int warmup, int iters)
{
    auto secs = measureFastestWithWarmup(fn, warmup, iters);

    double gflops = (1.0 * flops) / 1000000000;
    std::cout << "gflops: " << gflops << std::endl;
    std::cout << "GFLOPS: " << (gflops / secs) << std::endl;
}

int main()
{
    using facebook::sysml::aot::avx2;
    const int vector_size = std::is_same_v<CT_ISA, avx2> ? 8 : 16;

    {
        std::cout << "Toy example" << std::endl;
        const int ArCr = 32;
        const int AcBr = 32;
        const int BcCc = 32;

        auto A_vec = getRandomVector<float>(AcBr * ArCr);
        auto B_vec = getRandomVector<float>(AcBr * BcCc);
        auto CN    = getRandomVector<float>(ArCr * BcCc);

        // Start Halide definitions
        Halide::Buffer<float> A(A_vec.data(), {AcBr, ArCr}, "A");
        Halide::Buffer<float> B(B_vec.data(), {BcCc, AcBr}, "B");
        Halide::Func          C("C");

        Halide::Var arcr("arcr"), bccc("bccc");

        Halide::RDom acbr(0, AcBr, "acbr");
        Halide::RVar acbr_i("acbr_i");

        C(bccc, arcr) = 0.0f;
        C(bccc, arcr) += A(acbr, arcr) * B(bccc, acbr);

        C.update(0)
            .split(acbr, acbr, acbr_i, vector_size)
            .reorder(acbr_i, acbr, bccc, arcr)
            // added to force schedule, since + is associative
            .atomic(true)
            .vectorize(acbr_i)
            .unroll(acbr);

        C.bound(bccc, 0, BcCc).bound(arcr, 0, ArCr);

        C.print_loop_nest();
        C.compile_jit();

        Halide::Buffer<float> C_result = C.realize(B.width(), A.height());
        // End Halide definitions

        baseline_MM(ArCr, AcBr, BcCc, 1, ArCr, 1, AcBr, 1, ArCr, A.data(),
                    B.data(), CN.data(), 0);

        check_equivalence(C_result, CN.data());
        std::int64_t flops = AcBr * ArCr * BcCc * 2;
        benchmarking_stats([&]() { C.realize(B.width(), A.height()); }, flops,
                           10, 1000);
    }

    {
        std::cout << "Example 1: loop_nest.cpp (Matrix-Matrix product)"
                  << std::endl;
        const int ArCr = 256;
        const int AcBr = 256;
        const int BcCc = 256;

        auto A_vec = getRandomVector<float>(AcBr * ArCr);
        auto B_vec = getRandomVector<float>(AcBr * BcCc);
        auto CN    = getRandomVector<float>(ArCr * BcCc);

        // Start Halide definitions
        Halide::Buffer<float> A(A_vec.data(), {AcBr, ArCr}, "A");
        Halide::Buffer<float> B(B_vec.data(), {BcCc, AcBr}, "B");
        Halide::Func          C("C");

        Halide::Var arcr("arcr"), arcr_i("arcr_i");
        Halide::Var bccc("bccc"), bccc_i("bccc_o"), bccc_ii("bccc_ii");

        Halide::RDom acbr(0, AcBr);
        Halide::RVar acbr_o("acbr_o"), acbr_i("acbr_i");

        C(bccc, arcr) = 0.0f;
        C(bccc, arcr) += A(acbr, arcr) * B(bccc, acbr);

        C.store_in(Halide::MemoryType::Register)
            .update(0)
            .split(acbr, acbr, acbr_i, 256)
            .split(arcr, arcr, arcr_i, 3, Halide::TailStrategy::GuardWithIf)
            .split(bccc, bccc, bccc_i, 16)
            .split(bccc_i, bccc_i, bccc_ii, vector_size)
            .vectorize(bccc_ii)
            .unroll(bccc_i)
            .unroll(arcr_i)
            .reorder(bccc_ii, bccc_i, arcr_i, acbr_i, bccc, arcr, acbr);

        C.bound(bccc, 0, BcCc).bound(arcr, 0, ArCr);

        C.print_loop_nest();
        C.compile_jit();

        Halide::Buffer<float> C_result = C.realize(B.width(), A.height());
        // End Halide definitions

        baseline_MM(ArCr, AcBr, BcCc, 1, ArCr, 1, AcBr, 1, ArCr, A.data(),
                    B.data(), CN.data(), 0);

        check_equivalence(C_result, CN.data());
        std::int64_t flops = AcBr * ArCr * BcCc * 2;
        benchmarking_stats([&]() { C.realize(B.width(), A.height()); }, flops,
                           10, 1000);
    }

    // Convolution
    {
        std::cout << "Example 2: loop_nest.cpp (2D convolution on NCHW16c)"
                  << std::endl;
        const int GIN  = 128 / 16;
        const int CIN  = 16;
        const int GOUT = 128 / 16;
        const int COUT = 16;
        const int OS   = 56;
        const int KS   = 3;
        const int IS   = OS + KS - 1;

        auto A_vec = getRandomVector<float>(GIN * CIN * IS * IS);
        auto B_vec = getRandomVector<float>(GOUT * GIN * COUT * CIN * KS * KS);
        auto CN    = std::vector<float>(GOUT * COUT * OS * OS);

        Halide::Buffer<float> A(A_vec.data(), GIN * CIN * IS * IS);
        Halide::Buffer<float> B(B_vec.data(),
                                GOUT * GIN * CIN * COUT * KS * KS);
        Halide::Func          C("C");

        Halide::Var g_out("g_out");
        Halide::Var c_out("c_out"), c_out_i("c_out_i");
        Halide::Var o_h("o_h");
        Halide::Var o_w("o_w"), o_w_o("o_w_o"), o_w_i("o_w_i");

        // g_in: x, c_in: y, k_h: z, k_w: w
        Halide::RDom r({{0, GIN}, {0, CIN}, {0, KS}, {0, KS}}, "r");

        C(c_out, o_w, o_h, g_out) = 0.0f;
        C(c_out, o_w, o_h, g_out) +=
            A(r.x * IS * IS * CIN + o_h * IS * CIN + r.z * IS * CIN +
              o_w * CIN + r.w * CIN + r.y) *
            B(r.x * COUT * KS * KS * CIN * GOUT + g_out * COUT * KS * KS * CIN +
              r.y * COUT * KS * KS + r.z * COUT * KS + r.w * COUT + c_out);

        C.store_in(Halide::MemoryType::Register)
            .update(0)
            .split(o_w, o_w, o_w_i, 28, Halide::TailStrategy::GuardWithIf)
            .split(c_out, c_out, c_out_i, vector_size)
            .vectorize(c_out_i)
            .unroll(c_out)
            .unroll(r.w)
            .unroll(r.z)
            .reorder(c_out_i, c_out, r.w, r.z, o_w_i, r.y, r.x, o_h, o_w,
                     g_out);

        C.bound(o_w, 0, OS)
            .bound(o_h, 0, OS)
            .bound(c_out, 0, COUT)
            .bound(g_out, 0, GOUT);

        C.print_loop_nest();
        C.compile_jit();

        // dumping assembly for this example given slow-down on AVX2
        C.compile_to_assembly("out.asm", {}, "FunctionName");

        Halide::Buffer<float> C_result = C.realize(COUT, OS, OS, GOUT);

        baseline_Conv_NCHW8c(GOUT, COUT, GIN, CIN, OS, OS, KS, KS, A.data(),
                             B.data(), CN.data());

        check_equivalence(C_result, CN.data());

        std::int64_t flops = 2.0 * GIN * GOUT * CIN * COUT * OS * OS * KS * KS;
        benchmarking_stats([&]() { C.realize(COUT, OS, OS, GOUT); }, flops, 1,
                           100);
    }

    {
        const int ArCr = 1;
        const int AcBr = 333;
        const int BcCc = 333;

        float A_vec = 1.0;
        auto  B_vec = getRandomVector<float>(AcBr * BcCc);
        auto  CN    = getRandomVector<float>(ArCr * BcCc);

        // Start Halide definitions
        Halide::Buffer<float> A(&A_vec, {1, 1});
        Halide::Buffer<float> B(B_vec.data(), {BcCc, AcBr});
        Halide::Func          C("C");

        Halide::Var bccc("bccc"), bccc_i("bccc_i"), bccc_ii("bccc_ii");
        Halide::Var arcr("arcr");

        // A_{cols}: x, A_{rows}: y, AcBr: z
        Halide::RDom r({{0, 1}, {0, 1}, {0, AcBr}}, "r");
        Halide::RVar acbr_o("acbr_o"), acbr_i("acbr_i");

        C(bccc, arcr) = 0.0f;
        C(bccc, arcr) += A(r.x, r.y) * B(bccc, r.z);

        C.store_in(Halide::MemoryType::Register)
            .update(0)
            .split(r.z, acbr_o, acbr_i, 512, Halide::TailStrategy::GuardWithIf)
            .split(bccc, bccc, bccc_i, vector_size * 10,
                   Halide::TailStrategy::GuardWithIf)
            .split(bccc_i, bccc_i, bccc_ii, vector_size,
                   Halide::TailStrategy::GuardWithIf)
            .vectorize(bccc_ii)
            .unroll(bccc_i)
            .reorder(bccc_ii, bccc_i, bccc, acbr_i, acbr_o);

        C.print_loop_nest();
        C.bound(bccc, 0, BcCc).bound(arcr, 0, ArCr);
        C.compile_jit();

        Halide::Buffer<float> C_result = C.realize(BcCc, 1);

        baseline_MM(ArCr, AcBr, BcCc, 0, 0, BcCc, 1, BcCc, 1, &A_vec,
                    B_vec.data(), CN.data(), 0);
        check_equivalence(C_result, CN.data());

        std::int64_t flops = 1 * AcBr * BcCc * 2;
        benchmarking_stats([&]() { C.realize(BcCc, 1); }, flops, 10, 100);
    }
}
