// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <vector>

#include <sysml/measure.hpp>

#include "dabun/check.hpp"
#include "dabun/random_vector.hpp"
#include "dabun/tensillica/loop_nest.hpp"

#include "baselines.hpp"

int main()
{
    using namespace dabun;

    // Playing with weird schedules
    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 324;
        int AcBr = 124;
        int BcCc = 54;

        auto gen_loop_nest = [&]()
        {
            return dabun::tensillica::loop_nest_code_generator(
                       // The first argument is the loop order in the form of
                       // {dimension, stride}.  For now the outer dimension
                       // has to divide the stride.  This is effectively the
                       // same as Halide's split into outer and inner
                       // variable, but can have arbitray number of splits.
                       { {"ArCr", 6},
                         {"BcCc", 16},
                         {"AcBr", 4},
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, nullptr)
                .get_shared();
        };

        auto compile_secs = sysml::measure_fastest(gen_loop_nest, 1);
        std::cout << "Compile: " << compile_secs << std::endl;

        auto fn = gen_loop_nest();
        // fn.save_to_file("zi.asm");
        // fn.register_perf("fn1");

        auto A = get_random_vector<float>(AcBr * ArCr + 1024);
        auto B = get_random_vector<float>(AcBr * BcCc + 1024);

        auto CN = get_random_vector<float>(ArCr * BcCc + 1024);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1);
        // apply_relu(CN.data(), CN.data() + CN.size());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";

        auto secs = sysml::measure_fastest(
            [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";

        // bench_implementation_fmas_per_cycle(
        //     fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc,
        //     1.0 * AcBr * ArCr * BcCc * 2, 10, 10);
    }
}
