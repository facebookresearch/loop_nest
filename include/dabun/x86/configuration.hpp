// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"
#ifdef DABUN_ARCH_X86_64

namespace dabun
{
namespace x86
{

class OptimizationConfiguration
{
private:
    bool delay_innermost_operations_;
    bool split_vector_registers_;
    bool use_address_packer_;

public:
    OptimizationConfiguration(bool delay_innermost_operations,
                              bool split_vector_registers,
                              bool use_address_packer)
        : delay_innermost_operations_(delay_innermost_operations)
        , split_vector_registers_(split_vector_registers)
        , use_address_packer_(use_address_packer)

    {
    }

    OptimizationConfiguration()
        : delay_innermost_operations_(true)
        , split_vector_registers_(true)
        , use_address_packer_(true)
    {
    }

    bool delay_innermost_operations() const
    {
        return delay_innermost_operations_;
    }

    bool split_vector_registers() const { return split_vector_registers_; }

    bool use_address_packer() const { return use_address_packer_; }
};

inline OptimizationConfiguration all_optims(true, true, true);

// technically no optimizations beyond output tensor register blocking
inline OptimizationConfiguration no_optims(false, false, false);

} // namespace x86
} // namespace dabun

#endif
