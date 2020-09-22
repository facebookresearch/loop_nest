#include "dabun/check.hpp"
#include "dabun/measure.hpp"
#include "dabun/random_vector.hpp"
#include "dabun/transposer.hpp"
#include "transposer_baseline.hpp"

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

int main()
{
    using namespace dabun;

    srand(0);

    for (int rounds = 0; rounds < 1000000; ++rounds)
    {
        int ArCr         = (1 << rand() % 2) + rand() % 16;
        int AcBr         = (1 << rand() % 2) + rand() % 16;
        int max_unrolled = 1 << (rand() % 8);

        std::vector<std::pair<std::string, int>> order = {{"AcBr", 1},
                                                          {"ArCr", 1}};

        std::vector<std::pair<std::string, int>> hyper_order = {
            {"AcBr", (rand() % AcBr) + 2},  // It's OK to go oversize
                                            // (tests whether it's
                                            // handled appropriately)
            {"ArCr", (rand() % ArCr) + 2},  // - || -
            {"AcBr", (rand() % AcBr) + 2},  // - || -
            {"ArCr", (rand() % ArCr) + 2}}; // - || -

        std::sort(hyper_order.begin(), hyper_order.end(),
                  [](auto a, auto b) { return a.second > b.second; });

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
                        o.second = round_up(o.second,
                                            isa_traits<DABUN_ISA>::vector_size);
                    }
                }
                std::cout << o.first << "=" << o.second << "  ";
            }

            std::cout << "ArCr=" << ArCr << " ";
            std::cout << "AcBr=" << AcBr << " ";

            std::cout << "MU=" << max_unrolled << std::endl;

            auto fn_baselome =
                transposer_baseline(full_order, // The second argument is a
                                                // map of the dimension sizes
                                    {{"AcBr", AcBr}, {"ArCr", ArCr}},
                                    // out's strides for each variable.
                                    {{"ArCr", AcBr}, {"AcBr", 1}},
                                    // in's strides for each variable
                                    {{"ArCr", 1}, {"AcBr", ArCr}});

            auto fn = transposer_code_generator<DABUN_ISA>(
                          full_order, // The second argument is a map of the
                                      // dimension sizes
                          {{"AcBr", AcBr}, {"ArCr", ArCr}},
                          // out's strides for each variable.
                          {{"ArCr", AcBr}, {"AcBr", 1}},
                          // in's strides for each variable
                          {{"ArCr", 1}, {"AcBr", ArCr}}, max_unrolled)
                          .get_shared();

            fn.save_to_file("zi.asm");

            auto A  = get_random_vector<float>(AcBr * ArCr);
            auto CN = get_random_vector<float>(ArCr * AcBr);
            auto CJ = CN;

            fn_baselome(CN.data(), A.data());
            fn(CJ.data(), A.data());

            auto madiff = max_abs_difference(CJ.data(), CJ.data() + ArCr * AcBr,
                                             CN.data());

            std::cout << "ArCr=" << ArCr << " ";
            std::cout << "AcBr=" << AcBr << " ";

            std::cout << "ORDER: ";

            for (auto const& o : full_order)
            {
                std::cout << o.first << ',' << o.second << " :: ";
            }

            std::cout << "\n";

            std::cout << "MU=" << max_unrolled << std::endl;
            std::cout << "MAXABSDIFF: " << madiff << std::endl;

            assert(madiff < 0.000001);
        }
    }
}
