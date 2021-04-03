// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/code_generator.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace dabun
{
namespace x86
{

// Assumptions (to be assured by the wrapper):
//
// 0. [To be relaxed] All dimensions, and loop strides other than the innermost
//    three loops, are either divisible by 16 (64 bytes)
// 1. The three innermost loops have stride == 1 (let's name the dimensions K,
//    N, M starting from the innermost).
// 2. The innermost loop (K) is over the reduction variable.
// 3. The A tensor is packed along the reduction variable (K).
// 4. The 2nd from the last loop is over a dimension (N) that is packed in both
//    B and C.
// 5. There exist a loop (including implicit loops over full sizes), such that
//    the inner sizes of of M and N have such sizes so that a matrix of size MxN
//    fits in at most 6 of the accelerators Tmm register-matrices.

template <class AType, class BType = AType>
class amx_loop_nest_code_generator_impl
    : public code_generator<void(void* C, void const* A, void const* B,
                                 int alpha)>
{
private:
    using base =
        code_generator<void(float* C, float const* A, float const* B, int)>;

public:
    amx_loop_nest_code_generator_impl(
        std::vector<std::pair<std::string, int>> const& _order,
        std::map<std::string, int> const&               sizes,
        std::set<std::string> const&                    C_formula,
        std::set<std::string> const&                    A_formula,
        std::set<std::string> const&                    B_formula,
        std::map<std::string, int> const&               C_strides,
        std::map<std::string, int> const&               A_strides,
        std::map<std::string, int> const&               B_strides,
        std::optional<int> user_fma_unroll_limit = std::nullopt)
    {
    }
};

} // namespace x86
} // namespace dabun
