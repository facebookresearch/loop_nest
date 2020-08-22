// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#if !defined(ARM_LOOP_NEST)

#include "arithmetic_operation.h"

#else

#include <memory>

namespace facebook
{
namespace sysml
{
namespace aot
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

} // namespace aot
} // namespace sysml
} // namespace facebook

#endif
