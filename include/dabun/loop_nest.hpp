// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/float.hpp"
#include "dabun/isa.hpp"
#include "dabun/namespace.hpp"

#if defined(DABUN_ARM)
#include "dabun/arm/loop_nest.hpp"
#include "dabun/arm/loop_nest_fp16.hpp"
#else
#include "dabun/x86/loop_nest.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::loop_nest_code_generator;

#if defined(DABUN_ARM)

using DABUN_ISA_NAMESPACE ::loop_nest_fp16_code_generator;

template <extension VEX, class Arithmetic>
using loop_nest_compiler = std::conditional_t<
    std::is_same_v<Arithmetic, float>,
    loop_nest_code_generator<extension_to_deprecated_ISA_t<VEX>>,
    loop_nest_fp16_code_generator<extension_to_deprecated_ISA_t<VEX>>>;

#else

template <extension VEX, class Arithmetic>
using loop_nest_compiler = std::conditional_t<
    std::is_same_v<Arithmetic, float>,
    loop_nest_code_generator<extension_to_deprecated_ISA_t<VEX>>, void>;

#endif

} // namespace dabun
