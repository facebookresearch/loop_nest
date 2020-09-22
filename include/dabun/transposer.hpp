// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/namespace.hpp"

#if defined(DABUN_ARM)
#include "dabun/arm/transposer.hpp"
#else
#include "dabun/x86/transposer.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::transposer_code_generator;

} // namespace dabun
