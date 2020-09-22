// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>

namespace dabun
{
namespace arm
{

class operation_pair_base
{
};

inline std::shared_ptr<operation_pair_base> const fma =
    std::make_shared<operation_pair_base>();

// exclusively here to test non-fused operations as base case
inline std::shared_ptr<operation_pair_base> const non_fused_ma =
    std::make_shared<operation_pair_base>();

inline std::shared_ptr<operation_pair_base> const multiply_max =
    std::make_shared<operation_pair_base>();

inline std::shared_ptr<operation_pair_base> const multiply_min =
    std::make_shared<operation_pair_base>();

inline std::shared_ptr<operation_pair_base> const plus_max =
    std::make_shared<operation_pair_base>();

} // namespace arm
} // namespace dabun
