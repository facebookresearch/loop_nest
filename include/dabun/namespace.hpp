// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"

#if defined(DABUN_ARCH_AARCH64)
#define DABUN_ISA_NAMESPACE arm
#else
#define DABUN_ISA_NAMESPACE x86
#endif
