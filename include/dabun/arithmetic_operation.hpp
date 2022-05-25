// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"

#if defined(DABUN_ARCH_AARCH64)
#include "dabun/arm/arithmetic_operation.hpp"
#else
#include "dabun/x86/arithmetic_operation.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::fma;
using DABUN_ISA_NAMESPACE ::multiply_max;
using DABUN_ISA_NAMESPACE ::multiply_min;
using DABUN_ISA_NAMESPACE ::non_fused_ma;
using DABUN_ISA_NAMESPACE ::plus_max;

using DABUN_ISA_NAMESPACE ::operation_pair;
using DABUN_ISA_NAMESPACE ::operation_pair_base;

namespace op
{
using DABUN_ISA_NAMESPACE ::basic_multiplies;
using DABUN_ISA_NAMESPACE ::basic_plus;
using DABUN_ISA_NAMESPACE ::duplicate_base_plus;
using DABUN_ISA_NAMESPACE ::max;
using DABUN_ISA_NAMESPACE ::min;
} // namespace op

} // namespace dabun
