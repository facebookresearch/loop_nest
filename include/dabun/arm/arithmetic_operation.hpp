// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

namespace dabun
{
namespace arm
{

class operation_pair_base
{
};

template <class PlusType, class MultipliesType>
class operation_pair : public operation_pair_base
{
};

class basic_plus
{
};

class duplicate_base_plus
{
};

class max
{
};

class min
{
};

class basic_multiplies
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
