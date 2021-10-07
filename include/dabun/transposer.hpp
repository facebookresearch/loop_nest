// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"
#include "dabun/namespace.hpp"

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
