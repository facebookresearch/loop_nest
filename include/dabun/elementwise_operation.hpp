// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"

#if defined(DABUN_ARCH_AARCH64)
#include "dabun/arm/elementwise_operation.hpp"
#else
#include "dabun/x86/elementwise_operation.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::elementwise_bias;
using DABUN_ISA_NAMESPACE ::elementwise_multiply;
using DABUN_ISA_NAMESPACE ::elementwise_relu;

using DABUN_ISA_NAMESPACE ::elementwise_operation;
using DABUN_ISA_NAMESPACE ::relu_elementwise_operation;
using DABUN_ISA_NAMESPACE ::single_tensor_elementwise_operation;

} // namespace dabun
