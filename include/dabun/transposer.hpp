// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"

#if defined(DABUN_ARCH_AARCH64)
#include "dabun/arm/transposer.hpp"
#else
#include "dabun/x86/transposer.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::transposer_code_generator;

#if defined(DABUN_ARCH_AARCH64)

template <extension VEX, class Arithmetic>
using transposer_compiler =
    transposer_code_generator<extension_to_deprecated_ISA_t<VEX>, Arithmetic>;

#else

template <extension VEX, class Arithmetic>
using transposer_compiler =
    transposer_code_generator<extension_to_deprecated_ISA_t<VEX>>;

#endif

} // namespace dabun
