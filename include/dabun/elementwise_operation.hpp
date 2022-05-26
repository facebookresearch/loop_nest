// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"

#if defined(DABUN_ARCH_AARCH64)
#    include "dabun/arm/elementwise_operation.hpp"
#else
#    include "dabun/x86/elementwise_operation.hpp"
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
