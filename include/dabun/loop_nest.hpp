// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

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

} // namespace dabun
