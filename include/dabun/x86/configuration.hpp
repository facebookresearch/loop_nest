// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

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
