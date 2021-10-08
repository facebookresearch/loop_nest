// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"

#ifdef DABUN_ARCH_X86_64

#if !defined(XBYAK_NO_OP_NAMES)
#define XBYAK_NO_OP_NAMES
#endif

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

using xbyak_buffer_type = Xbyak::uint8;

#endif
