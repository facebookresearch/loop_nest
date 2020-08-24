#include "loop_nest.h"

#include "baselines.h"
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
    using facebook::sysml::aot::aarch64;
    using facebook::sysml::aot::avx2;
    using facebook::sysml::aot::avx2_plus;
    using facebook::sysml::aot::avx512;

    for (int rounds = 0; rounds < 1000000; ++rounds)
    {
        int ArCr              = (1 << rand() % 10) + rand() % 16;
        int AcBr              = (1 << rand() % 10) + rand() % 16;
        int BcCc              = (1 << rand() % 10) + rand() % 16;
        int max_fmas_unrolled = 1 << (rand() % 16);

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
                            o.second = facebook::sysml::aot::round_up(
                                o.second, facebook::sysml::aot::isa_traits<
                                              CT_ISA>::vector_size);
                        }
                    }
                    std::cout << o.first << "=" << o.second << "  ";
                }

                std::cout << "ArCr=" << ArCr << " ";
                std::cout << "AcBr=" << AcBr << " ";
                std::cout << "BcCc=" << BcCc << " ";

                std::cout << "MU=" << max_fmas_unrolled << std::endl;

                auto fn =
                    facebook::sysml::aot::FMA_loop_nest_jitter<CT_ISA>(
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
                        {{"AcBr", BcCc}, {"BcCc", 1}}, nullptr,
                        max_fmas_unrolled,
                        nullptr)
                        .get_shared();

                auto A = getRandomVector<float>(AcBr * ArCr);
                auto B = getRandomVector<float>(AcBr * BcCc);

                auto CN = getRandomVector<float>(ArCr * BcCc);
                auto CJ = CN;

                baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1,
                            A.data(), B.data(), CN.data(), 1);

                // apply_relu(CN.data(), CN.data() + CN.size());

                fn(CJ.data(), A.data(), B.data(), 1);

                auto madiff =
                    maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data());

                std::cout << "MAXABSDIFF: " << madiff << std::endl;

                // assert(madiff < 0.001);
            } while (std::next_permutation(order.begin(), order.end()));
        } while (std::next_permutation(hyper_order.begin(), hyper_order.end()));
    }
}
