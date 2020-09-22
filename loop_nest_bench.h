#pragma once

#include "loop_nest.h"
#include "utils.h"

#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace facebook
{
namespace sysml
{
namespace aot
{

template <class ISA>
void loop_nest_bench(std::vector<std::pair<std::string, int>> const& order,
                     std::map<std::string, int> const&               sizes,
                     std::set<std::string> const&                    C_formula,
                     std::set<std::string> const&                    A_formula,
                     std::set<std::string> const&                    B_formula,
                     std::map<std::string, int> const&               C_strides,
                     std::map<std::string, int> const&               A_strides,
                     std::map<std::string, int> const& B_strides, int alpha = 0,
                     int max_unrolled_fmas = 320, int total_iterations = 100,
                     int warmup_iterations = 10)
{
    std::int64_t C_size = 1;
    std::int64_t A_size = 1;
    std::int64_t B_size = 1;

    double flops = 2.0;

    for (auto const& s : sizes)
    {
        if (C_strides.count(s.first))
            C_size += (s.second - 1) * C_strides.at(s.first);
        if (A_strides.count(s.first))
            A_size += (s.second - 1) * A_strides.at(s.first);
        if (B_strides.count(s.first))
            B_size += (s.second - 1) * B_strides.at(s.first);
        if (C_strides.count(s.first) || B_strides.count(s.first) ||
            A_strides.count(s.first))
            flops *= s.second;
    }

    auto A  = get_random_vector<float>(A_size);
    auto B  = get_random_vector<float>(B_size);
    auto CN = get_random_vector<float>(C_size);

    auto jit_fn = loop_nest_code_generator<ISA>(order, sizes, C_formula, A_formula,
                                            B_formula, C_strides, A_strides,
                                            B_strides, max_unrolled_fmas)
                      .get_shared();

    jit_fn.save_to_file("zi.asm");

    auto secs = measureFastestWithWarmup(
        [&]() { jit_fn(CN.data(), A.data(), B.data(), alpha); },
        warmup_iterations, total_iterations);

    double gflops = flops / 1000000000;

    std::cout << "gflops: " << gflops << "\n";

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
}

} // namespace aot
} // namespace sysml
} // namespace facebook
