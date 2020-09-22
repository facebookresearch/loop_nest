// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/namespace.hpp"

#if defined(DABUN_ARM)
#include "dabun/arm/elementwise_operation.hpp"
#else
#include "dabun/x86/elementwise_operation.hpp"
#endif

namespace dabun
{

using DABUN_ISA_NAMESPACE ::elementwise_bias;
using DABUN_ISA_NAMESPACE ::elementwise_multiply;
using DABUN_ISA_NAMESPACE ::elementwise_relu;

} // namespace dabun
