// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"
#include "dabun/predef.hpp"

#ifdef DABUN_ARCH_X86_64

#    if DABUN_COMP_GNUC
#        if DABUN_COMP_GNUC >= DABUN_VERSION_NUMBER(11, 0, 0)
#            define DABUN_XBYAK_SUPRESS_WARRAY_BOUNDS
#        endif
#    endif

#    if !defined(XBYAK_NO_OP_NAMES)
#        define XBYAK_NO_OP_NAMES
#    endif

#    ifdef DABUN_XBYAK_SUPRESS_WARRAY_BOUNDS
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Warray-bounds"
#    endif

#    include "xbyak/xbyak.h"
#    include "xbyak/xbyak_util.h"

#    ifdef DABUN_XBYAK_SUPRESS_WARRAY_BOUNDS
#        pragma GCC diagnostic pop
#        undef DABUN_XBYAK_SUPRESS_WARRAY_BOUNDS
#    endif

using xbyak_buffer_type = Xbyak::uint8;

#endif
