// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"
#include "dabun/predef.hpp"

#ifdef DABUN_ARCH_X86_64

#if BOOST_COMP_GNUC
  #if BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(11,0,0)
    #define DABUN_XBYAK_SUPRESS_WARRAY_BOUNDS
  #endif
#endif

#if !defined(XBYAK_NO_OP_NAMES)
#define XBYAK_NO_OP_NAMES
#endif

#ifdef DABUN_XBYAK_SUPRESS_WARRAY_BOUNDS
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"


#ifdef DABUN_XBYAK_SUPRESS_WARRAY_BOUNDS
  #pragma GCC diagnostic pop
  #undef DABUN_XBYAK_SUPRESS_WARRAY_BOUNDS
#endif


using xbyak_buffer_type = Xbyak::uint8;

#endif
