// Copyright 2004-present Facebook. All Rights Reserved.

#include "AlignedVec.h"
#include "baselines.h"
#include "loop_nest.h"
#include "loop_nest_baseline.h"
#include "one_constant.h"
#include "translate_to_halide.h"
#include "utils.h"

#ifndef CT_ISA
#define CT_ISA avx2
#endif

int main() {
  using facebook::sysml::aot::aot_fn_cast;
  using facebook::sysml::aot::avx2;
  using facebook::sysml::aot::avx2_plus;
  using facebook::sysml::aot::avx512;

  {
    std::cout << "Benchmark: GEMM-256" << std::endl;

    int ArCr = 256;
    int AcBr = 256;
    int BcCc = 256;

    auto fn = facebook::sysml::aot::LoopNestToHalide<CT_ISA>(
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
        // to be ordered though.. Many decisions to make..
        {{"ArCr", BcCc}, {"BcCc", 1}},
        // A's strides for each variable
        {{"ArCr", AcBr}, {"AcBr", 1}},
        // B's strides for each variable
        {{"AcBr", BcCc}, {"BcCc", 1}}, 1024);

    auto A = getRandomVector<float>(AcBr * ArCr);
    auto B = getRandomVector<float>(AcBr * BcCc);

    auto CN = getRandomVector<float>(ArCr * BcCc);
    auto CJ = CN;

    baseline_MM(ArCr, AcBr, BcCc, 1, ArCr, 1, AcBr, 1, ArCr, A.data(), B.data(),
                CN.data(), 0);

    auto compile = [&fn]() { fn.compile_jit(); };
    auto compile_secs = measureFastestWithWarmup(compile, 0, 1);
    std::cout << "Compile: " << compile_secs << std::endl;

    fn.run_on_aligned_data(CJ.data(), A.data(), B.data());

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
              << "\n";

    auto secs = measureFastestWithWarmup(
        [&]() { fn.run_on_aligned_data(CJ.data(), A.data(), B.data()); }, 10,
        1000);

    double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
  }

  {
    std::cout << "Benchmark: GEMM-512" << std::endl;

    int ArCr = 512;
    int AcBr = 512;
    int BcCc = 512;

    auto fn = facebook::sysml::aot::LoopNestToHalide<CT_ISA>(
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
        // to be ordered though.. Many decisions to make..
        {{"ArCr", BcCc}, {"BcCc", 1}},
        // A's strides for each variable
        {{"ArCr", AcBr}, {"AcBr", 1}},
        // B's strides for each variable
        {{"AcBr", BcCc}, {"BcCc", 1}}, 1024);

    auto A = getRandomVector<float>(AcBr * ArCr);
    auto B = getRandomVector<float>(AcBr * BcCc);

    auto CN = getRandomVector<float>(ArCr * BcCc);
    auto CJ = CN;

    baseline_MM(ArCr, AcBr, BcCc, 1, ArCr, 1, AcBr, 1, ArCr, A.data(), B.data(),
                CN.data(), 0);

    auto compile = [&fn]() { fn.compile_jit(); };
    auto compile_secs = measureFastestWithWarmup(compile, 0, 1);
    std::cout << "Compile: " << compile_secs << std::endl;

    fn.run_on_aligned_data(CJ.data(), A.data(), B.data());

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
              << "\n";

    auto secs = measureFastestWithWarmup(
        [&]() { fn.run_on_aligned_data(CJ.data(), A.data(), B.data()); }, 10,
        1000);

    double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
  }

  // (row-major)Matrix-(column)Vector product (requires horizontal
  // sum) C(r) = A(r, k) * B(k)
  {
    std::cout << "Benchmark: GEMV-256" << std::endl;

    int ArCr = 256;
    int AcBr = 256;
    int BcCc = 1;

    int k = AcBr;
    int r = ArCr;

    auto fn = facebook::sysml::aot::LoopNestToHalide<CT_ISA>(
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
        {{"k", 2}}, 1024);

    auto A = getRandomVector<float>(AcBr * ArCr * 2);
    auto B = getRandomVector<float>(AcBr * BcCc * 2);

    auto CN = getRandomVector<float>(ArCr * BcCc);
    auto CJ = CN;

    baseline_MM(ArCr, AcBr, BcCc, k * 2, 2, 2, 2, 1, 1, A.data(), B.data(),
                CN.data());

    auto compile = [&fn]() { fn.compile_jit(); };
    auto compile_secs = measureFastestWithWarmup(compile, 0, 1);
    std::cout << "Compile: " << compile_secs << std::endl;

    fn.run_on_aligned_data(CJ.data(), A.data(), B.data());

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
              << "\n";

    auto secs = measureFastestWithWarmup(
        [&]() { fn.run_on_aligned_data(CJ.data(), A.data(), B.data()); }, 10,
        10);

    double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
  }

  // (row) Vector - (row - major) Matrix product C(c) = A(k) * B(k, c)
  {
    std::cout << "Benchmark: GEVM-256" << std::endl;

    int ArCr = 1;
    int AcBr = 256;
    int BcCc = 256;

    int k = AcBr;
    int c = BcCc;

    auto fn = facebook::sysml::aot::LoopNestToHalide<CT_ISA>(
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

    auto A = getRandomVector<float>(AcBr * ArCr);
    auto B = getRandomVector<float>(AcBr * BcCc);

    auto CN = getRandomVector<float>(ArCr * BcCc);
    auto CJ = CN;

    baseline_MM(ArCr, AcBr, BcCc, 0, 1, c, 1, 0, 1, A.data(), B.data(),
                CN.data(), 0);

    auto compile = [&fn]() { fn.compile_jit(); };
    auto compile_secs = measureFastestWithWarmup(compile, 0, 1);
    std::cout << "Compile: " << compile_secs << std::endl;

    fn.run_on_aligned_data(CJ.data(), A.data(), B.data());

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
              << "\n";

    auto secs = measureFastestWithWarmup(
        [&]() { fn.run_on_aligned_data(CJ.data(), A.data(), B.data()); }, 10,
        10);

    double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
  }

  // 2D convolution on NCHW16c layout example:
  // O(g_out, c_out, o_h, o_w) = I(g_in, c_in, o_h + k_h, ow + k_w) *
  //                             K(g_in, g_out, c_in, c_out, k_h, k_w)
  {
    std::cout << "Benchmark: 2D-Conv" << std::endl;

    int GIN = 128 / 16;
    int CIN = 16;
    int GOUT = 128 / 16;
    int COUT = 16;
    int OS = 56;
    int KS = 3;
    int IS = OS + KS - 1;

    auto fn = facebook::sysml::aot::LoopNestToHalide<CT_ISA>(
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
         {"i_w", IS},
         {"i_h", IS}},
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

    auto A = getRandomVector<float>(GIN * CIN * IS * IS);
    auto B = getRandomVector<float>(GOUT * GIN * COUT * CIN * KS * KS);
    auto CN = getRandomVector<float>(GOUT * COUT * OS * OS);
    auto CJ = CN;

    baseline_Conv_NCHW8c(GOUT, COUT, GIN, CIN, OS, OS, KS, KS, A.data(),
                         B.data(), CN.data());

    auto compile = [&fn]() { fn.compile_jit(); };
    auto compile_secs = measureFastestWithWarmup(compile, 0, 1);
    std::cout << "Compile: " << compile_secs << std::endl;

    fn.run_on_aligned_data(CJ.data(), A.data(), B.data());

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(CJ.data(), CJ.data() + COUT * OS * OS, CN.data())
              << "\n";

    auto secs = measureFastestWithWarmup(
        [&]() { fn.run_on_aligned_data(CJ.data(), A.data(), B.data()); }, 1,
        100);

    double gflops =
        2.0 * GIN * GOUT * CIN * COUT * OS * OS * KS * KS / 1000000000;

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
  }

  // Single 3D convolution
  // C(x, y, z) = A(x + x_k, y + y_k, z + z_k) *
  //              B(x_k, y_k, z_k)
  {
    std::cout << "Benchmark: 3D-Conv" << std::endl;

    int OX = 101;
    int OY = 101;
    int OZ = 16 * 12 + 3;
    int KX = 3;
    int KY = 3;
    int KZ = 3;
    int IX = OX + KX - 1;
    int IY = OY + KY - 1;
    int IZ = OZ + KZ - 1;

    auto fn = facebook::sysml::aot::LoopNestToHalide<CT_ISA>(
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
        {
            {"OX", OX},
            {"OY", OY},
            {"OZ", OZ},
            {"KX", KX},
            {"KY", KY},
            {"KZ", KZ},
            {"IX", IX},
            {"IY", IY},
            {"IZ", IZ},
        },
        // Vars of C (other variables are reduction variables)
        {"OX", "OY", "OZ"},
        // Variables of A
        {"IX", "IY", "IZ"},
        // Variables of B
        {"KX", "KY", "KZ"}, {{"OX", OY * OZ}, {"OY", OZ}, {"OZ", 1}},
        // A's strides for each variable
        {{"OX", IY * IZ},
         {"OY", IZ},
         {"OZ", 1},
         {"KX", IY * IZ},
         {"KY", IZ},
         {"KZ", 1}},
        // B's strides for each variable
        {{"KX", KY * KZ}, {"KY", KZ}, {"KZ", 1}}, 1024);

    auto A = getRandomVector<float>(IX * IY * IZ);
    auto B = getRandomVector<float>(KX * KY * KZ);

    auto CN = getRandomVector<float>(OX * OY * OZ);
    auto CJ = CN;

    baseline_3DConv(OX, OY, OZ, KX, KY, KZ, A.data(), B.data(), CN.data());

    auto compile = [&fn]() { fn.compile_jit(); };
    auto compile_secs = measureFastestWithWarmup(compile, 0, 1);
    std::cout << "Compile: " << compile_secs << std::endl;

    fn.run_on_aligned_data(CJ.data(), A.data(), B.data());

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(CJ.data(), CJ.data() + OX * OY * OZ, CN.data())
              << "\n";

    auto secs = measureFastestWithWarmup(
        [&]() { fn.run_on_aligned_data(CJ.data(), A.data(), B.data()); }, 1, 1);

    double gflops = 1.0 * OX * OY * OZ * KX * KY * KZ * 2 / 1000000000;

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
  }
}
