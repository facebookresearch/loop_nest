// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/namespace.hpp"

#if defined(DABUN_ARM)
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

} // namespace dabun
