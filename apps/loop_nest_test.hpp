#pragma once

#include "dabun/arithmetic_operation.hpp"
#include "dabun/check.hpp"
#include "dabun/loop_nest.hpp"
#include "dabun/measure.hpp"
#include "dabun/random_vector.hpp"
#include "loop_nest_baseline.hpp"

#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dabun
{

template <class ISA>
void test_loop_nest_against_slow_baseline(
    std::vector<std::pair<std::string, int>> const& order,
    std::map<std::string, int> const&               sizes,
    std::set<std::string> const&                    C_formula,
    std::set<std::string> const&                    A_formula,
    std::set<std::string> const&                    B_formula,
    std::map<std::string, int> const&               C_strides,
    std::map<std::string, int> const&               A_strides,
    std::map<std::string, int> const& B_strides, int max_unrolled_fmas = 512,
    int alpha = 1)
{
    std::int64_t C_size = 1;
    std::int64_t A_size = 1;
    std::int64_t B_size = 1;

    alpha = alpha ? 1 : 0;

    for (auto const& s : sizes)
    {
        if (C_strides.count(s.first))
            C_size += (s.second - 1) * C_strides.at(s.first);
        if (A_strides.count(s.first))
            A_size += (s.second - 1) * A_strides.at(s.first);
        if (B_strides.count(s.first))
            B_size += (s.second - 1) * B_strides.at(s.first);
    }

    auto A  = get_random_vector<float>(A_size);
    auto B  = get_random_vector<float>(B_size);
    auto CN = get_random_vector<float>(C_size);
    auto CJ = CN;

    auto jit_fn = loop_nest_code_generator<ISA>(
                      order, sizes, C_formula, A_formula, B_formula, C_strides,
                      A_strides, B_strides, fma, max_unrolled_fmas)
                      .get_shared();

    jit_fn.save_to_file("zi.asm");

    auto baseline_fn =
        loop_nest_baseline(order, sizes, C_formula, A_formula, B_formula,
                           C_strides, A_strides, B_strides, false);

    jit_fn(CJ.data(), A.data(), B.data(), alpha);
    baseline_fn(CN.data(), A.data(), B.data(), alpha);

    std::cout << "MAXABSDIFF: ( " << C_size << " ) "
              << max_abs_difference(CJ.data(), CJ.data() + C_size, CN.data())
              << "\n";
}

} // namespace dabun
