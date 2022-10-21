// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"
#include "dabun/numeric.hpp"

#if defined(DABUN_ARCH_AARCH64)
#    include "dabun/arm/loop_nest.hpp"
#else
#    include "dabun/x86/loop_nest.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::loop_nest_code_generator;

#if defined(DABUN_ARCH_AARCH64)

using DABUN_ISA_NAMESPACE ::loop_nest_fp16_code_generator;

template <extension VEX, class Arithmetic>
using loop_nest_compiler = std::conditional_t<
    std::is_same_v<Arithmetic, float>,
    loop_nest_code_generator<extension_to_deprecated_ISA_t<VEX>, false>,
    loop_nest_code_generator<extension_to_deprecated_ISA_t<VEX>, true>>;

#else

template <extension VEX, class Arithmetic>
using loop_nest_compiler = std::conditional_t<
    std::is_same_v<Arithmetic, float>,
    loop_nest_code_generator<extension_to_deprecated_ISA_t<VEX>>, void>;

#endif

} // namespace dabun
