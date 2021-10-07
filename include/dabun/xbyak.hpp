// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"

#if defined(DABUN_ARCH_AARCH64)

#include "dabun/arm/xbyak.hpp"

#else

#include "dabun/x86/xbyak.hpp"

#endif
