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

    using facebook::sysml::aot::elementwise_bias;
    using facebook::sysml::aot::elementwise_relu;

    {
        std::cout << "Example 1 (packed, no tail masking)" << std::endl;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024, nullptr, {},
                       compose(elementwise_bias<CT_ISA>,
                               elementwise_relu<CT_ISA>),
                       {
                           {{"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 1);
        auto CJ   = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        apply_relu(CN.data(), CN.data() + ArCr * BcCc);

        fn(CJ.data(), A.data(), B.data(), 0, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Example 2 (packed, with tail masking)" << std::endl;

        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 255;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024, nullptr, {},
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 1);
        auto CJ   = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());

        fn(CJ.data(), A.data(), B.data(), 0, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Example 3 (strided, with tail masking)" << std::endl;

        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 255;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024, nullptr, {},
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 3}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 3);
        auto CJ   = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 3, CN.data(), bias.data());

        fn(CJ.data(), A.data(), B.data(), 0, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Example 4 (scalar)" << std::endl;

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
                       {
                           {"ArCr", 1},
                           {"BcCc", 1},
                           {"AcBr", 1},
                       },
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 20, nullptr, {},
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 1);
        auto CJ   = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());

        fn(CJ.data(), A.data(), B.data(), 0, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Example 5 (scalar)" << std::endl;

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
                       {
                           {"BcCc", 1},
                           {"AcBr", 1},
                           {"ArCr", 1},
                       },
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 20, nullptr, {},
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"ArCr", 0}, {"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 1);
        auto CJ   = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());

        fn(CJ.data(), A.data(), B.data(), 0, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    // pre-op tests
    {
        std::cout << "Pre-Op Example 1 (packed, no tail masking)" << std::endl;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024,
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 1);
        auto CJ   = CN;

        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Preop Example 2 (packed, with tail masking)" << std::endl;

        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 255;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024,
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 1);
        auto CJ   = CN;

        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Preop Example 3 (strided, with tail masking)"
                  << std::endl;

        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 255;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024,
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 3}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 3);
        auto CJ   = CN;

        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 3, CN.data(), bias.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Preop Example 4 (scalar)" << std::endl;

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
                       {
                           {"ArCr", 1},
                           {"BcCc", 1},
                           {"AcBr", 1},
                       },
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 20,
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 1);
        auto CJ   = CN;

        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Preop Example 5 (scalar)" << std::endl;

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
                       {
                           {"BcCc", 1},
                           {"AcBr", 1},
                           {"ArCr", 1},
                       },
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 20,
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"ArCr", 0}, {"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));
        fn.save_to_file("zi.asm");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN   = getRandomVector<float>(ArCr * BcCc);
        auto bias = getRandomVector<float>(1 * BcCc * 1);
        auto CJ   = CN;

        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1, bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Pre-Op Example 6 (packed, no tail masking)" << std::endl;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024,
                       facebook::sysml::aot::elementwise_multiply<CT_ISA>,
                       {
                           {{"ArCr", 0}, {"BcCc", 0}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN    = getRandomVector<float>(ArCr * BcCc);
        auto other = std::vector<float>({-1.0});
        auto CJ    = CN;

        baseline_matrix_elementwise_multiply(ArCr, BcCc, BcCc, 1, 0, 0,
                                             CN.data(), other.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn(CJ.data(), A.data(), B.data(), 1, other.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Preop + PostOp (packed, with tail masking)" << std::endl;

        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 255;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024,
                       facebook::sysml::aot::elementwise_multiply<CT_ISA>,
                       {
                           {{"ArCr", 0}, {"BcCc", 0}},
                       },
                       facebook::sysml::aot::elementwise_relu<CT_ISA>)
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN    = getRandomVector<float>(ArCr * BcCc);
        auto other = std::vector<float>({-2.0});
        auto CJ    = CN;

        baseline_matrix_elementwise_multiply(ArCr, BcCc, BcCc, 1, 0, 0,
                                             CN.data(), other.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);
        apply_relu(CN.data(), CN.data() + ArCr * BcCc);

        fn(CJ.data(), A.data(), B.data(), 1, other.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Preop + PostOp Example 2 (strided, with tail masking)"
                  << std::endl;

        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 255;

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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 1024,
                       facebook::sysml::aot::elementwise_multiply<CT_ISA>,
                       {{{"ArCr", 0}, {"BcCc", 0}}},
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 3}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*, float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN    = getRandomVector<float>(ArCr * BcCc);
        auto other = std::vector<float>({-1.0});
        auto bias  = getRandomVector<float>(1 * BcCc * 3);
        auto CJ    = CN;

        baseline_matrix_elementwise_multiply(ArCr, BcCc, BcCc, 1, 0, 0,
                                             CN.data(), other.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 3, CN.data(), bias.data());

        fn(CJ.data(), A.data(), B.data(), 1, other.data(), bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {
        std::cout << "Preop + Post Op Example 4 (scalar)" << std::endl;

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
                       {
                           {"ArCr", 1},
                           {"BcCc", 1},
                           {"AcBr", 1},
                       },
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
                       {{"AcBr", BcCc}, {"BcCc", 1}}, 20,
                       facebook::sysml::aot::elementwise_multiply<CT_ISA>,
                       {{{"ArCr", 0}, {"BcCc", 0}}},
                       facebook::sysml::aot::elementwise_bias<CT_ISA>,
                       {
                           {{"BcCc", 1}},
                       })
                .get_unique();
        };

        auto fnx = gen_loop_nest();
        auto fny = aot_fn_cast<void(int)>(std::move(fnx));
        auto fn  = aot_fn_cast<void(float*, float const*, float const*, int,
                                   float const*, float const*)>(std::move(fny));

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN    = getRandomVector<float>(ArCr * BcCc);
        auto other = std::vector<float>({-1.0});
        auto bias  = getRandomVector<float>(1 * BcCc * 1);
        auto CJ    = CN;

        baseline_matrix_elementwise_multiply(ArCr, BcCc, BcCc, 1, 0, 0,
                                             CN.data(), other.data());
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());

        fn(CJ.data(), A.data(), B.data(), 1, other.data(), bias.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }
}
