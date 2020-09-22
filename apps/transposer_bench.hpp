#include "dabun/measure.hpp"
#include "dabun/transposer.hpp"

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

namespace dabun
{

template <class ISA>
void transposer_bench(std::vector<std::pair<std::string, int>> const& order,
                      std::map<std::string, int> const&               sizes,
                      std::map<std::string, int> const& out_strides,
                      std::map<std::string, int> const& in_strides,
                      int max_unrolled_fmas = 320, int total_iterations = 100)

{
    auto total_moved_bytes =
        std::accumulate(sizes.begin(), sizes.end(), 1,
                        [](auto a, auto b) { return a * b.second; }) *
        4;

    std::int64_t in_size  = 1;
    std::int64_t out_size = 1;

    for (auto const& s : sizes)
    {
        in_size += (s.second - 1) * in_strides.at(s.first);
        out_size += (s.second - 1) * out_strides.at(s.first);
    }

    auto A = get_random_vector<float>(in_size);
    auto B = get_random_vector<float>(out_size);

    auto jit_fn = transposer_code_generator<ISA>(order, sizes, out_strides,
                                                 in_strides, max_unrolled_fmas)
                      .get_unique();

    jit_fn.save_to_file("zi.asm");

    auto secs = measure_fastest([&]() { jit_fn(B.data(), A.data()); },
                                total_iterations);

    double moved_gbytes = 1.0 * total_moved_bytes / 1000000000;

    std::cout << "GBPS: " << (moved_gbytes / secs) << "\n";
    std::cout << "MSEC: " << (secs / 1000) << "\n";
}

} // namespace dabun
