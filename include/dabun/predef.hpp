// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// Inspired by boost predef

#define DABUN_VERSION_NUMBER(major, minor, patch)                              \
    ((((major) % 100) * 10000000) + (((minor) % 100) * 100000) +               \
     ((patch) % 100000))

#if defined(__clang__)

#    define DABUN_COMP_CLANG                                                   \
        DABUN_VERSION_NUMBER(__clang_major__, __clang_minor__,                 \
                             __clang_patchlevel__)

#elif defined(__GNUC__)

#    define DABUN_COMP_GNUC                                                    \
        DABUN_VERSION_NUMBER(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)

#else

#    error "Compiler not supported"

#endif
