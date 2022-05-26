// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"

#if defined(DABUN_ARCH_AARCH64)
#    include "dabun/arm/configuration.hpp"
#else
#    include "dabun/x86/configuration.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::all_optims;
using DABUN_ISA_NAMESPACE ::no_optims;
using DABUN_ISA_NAMESPACE ::OptimizationConfiguration;

} // namespace dabun
