// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"
#include "dabun/namespace.hpp"

#if defined(DABUN_ARCH_AARCH64)
#include "dabun/arm/configuration.hpp"
#else
#include "dabun/x86/configuration.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::all_optims;
using DABUN_ISA_NAMESPACE ::no_optims;
using DABUN_ISA_NAMESPACE ::OptimizationConfiguration;

} // namespace dabun