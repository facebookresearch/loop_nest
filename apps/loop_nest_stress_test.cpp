// Copyright 2004-present Facebook. All Rights Reserved.

#include "baselines.hpp"
#include "loop_nest_baseline.hpp"

#include "dabun/arithmetic_operation.hpp"
#include "dabun/check.hpp"
#include "dabun/isa.hpp"
#include "dabun/loop_nest.hpp"
#include "dabun/measure.hpp"
#include "dabun/random_vector.hpp"

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

#ifndef DABUN_ISA
#define DABUN_ISA avx2
#endif

#ifndef DABUN_ARITHMETIC
#define DABUN_ARITHMETIC float
#endif

#ifndef DABUN_VEX
#if defined(DABUN_ARCH_AARCH64)
#define DABUN_VEX ::dabun::extension::neon
#else
#define DABUN_VEX ::dabun::extension::avx2
#endif
#endif

int main()
{
    using namespace dabun;

    using float_t = DABUN_ARITHMETIC;

    for (int rounds = 0; rounds < 1000000; ++rounds)
    {
        int ArCr              = (1 << rand() % 10) + rand() % 16;
        int AcBr              = (1 << rand() % 10) + rand() % 16;
        int BcCc              = (1 << rand() % 10) + rand() % 16;
        int max_fmas_unrolled = 1 << (rand() % 10);

        std::vector<std::pair<std::string, int>> order = {
            {"AcBr", 1}, {"BcCc", 1}, {"ArCr", 1}};

        std::vector<std::pair<std::string, int>> hyper_order = {
            {"AcBr", (rand() % AcBr) + 2},  // It's OK to go oversize
                                            // (tests whether it's
                                            // handled appropriately)
            {"BcCc", (rand() % BcCc) + 2},  // - || -
            {"ArCr", (rand() % ArCr) + 2}}; // - || -

        std::sort(hyper_order.begin(), hyper_order.end());

        do
        {
            std::sort(order.begin(), order.end());
            do
            {
                auto full_order = hyper_order;
                full_order.insert(full_order.end(), order.begin(), order.end());

                std::cout << "DIF: ORDER: ";
                for (auto& o : full_order)
                {
                    if (o.first == full_order.back().first)
                    {
                        if (o.second != 1)
                        {
                            o.second = round_up(
                                o.second,
                                isa_traits<extension_to_deprecated_ISA_t<
                                        DABUN_VEX>>::vector_size *
                                    4 / sizeof(float_t));
                        }
                    }
                    std::cout << o.first << "=" << o.second << "  ";
                }

                std::cout << "ArCr=" << ArCr << " ";
                std::cout << "AcBr=" << AcBr << " ";
                std::cout << "BcCc=" << BcCc << " ";

                std::cout << "MU=" << max_fmas_unrolled << std::endl;

                auto fn =
                    loop_nest_compiler<DABUN_VEX, float_t>(
                        full_order, // The second argument is a map of the
                                    // dimension sizes
                        {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
                        // Vars of C (other variables are reduction variables)
                        {"ArCr", "BcCc"},
                        // Variables of A
                        {"ArCr", "AcBr"},
                        // Variables of B
                        {"AcBr", "BcCc"},
                        // C's strides for each variable.
                        {{"ArCr", BcCc}, {"BcCc", 1}},
                        // A's strides for each variable
                        {{"ArCr", AcBr}, {"AcBr", 1}},
                        // B's strides for each variable
                        {{"AcBr", BcCc}, {"BcCc", 1}}, dabun::fma,
                        max_fmas_unrolled, nullptr)
                        .get_shared();

                auto A = get_random_vector<float_t>(AcBr * ArCr);
                auto B = get_random_vector<float_t>(AcBr * BcCc);

                auto CN = get_random_vector<float_t>(ArCr * BcCc);
                auto CJ = CN;

                baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1,
                            A.data(), B.data(), CN.data(), 1);

                // apply_relu(CN.data(), CN.data() + CN.size());

                fn(CJ.data(), A.data(), B.data(), 1);

                auto madiff = max_abs_difference(
                    CJ.data(), CJ.data() + ArCr * BcCc, CN.data());

                std::cout << "MAXABSDIFF: " << madiff << std::endl;

                // assert(madiff < 0.001);
            } while (std::next_permutation(order.begin(), order.end()));
        } while (std::next_permutation(hyper_order.begin(), hyper_order.end()));
    }
}
