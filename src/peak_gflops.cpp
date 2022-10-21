// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef DABUN_HEADER_ONLY

#    include "dabun/peak_gflops.hpp"

namespace dabun
{

#    if defined(DABUN_ARCH_AARCH64)

namespace arm
{

template struct bench_gflops<aarch64, fp32_t>;
template struct bench_gflops<aarch64, fp16_t>;

} // namespace arm

#    else

namespace x86
{

template struct bench_gflops<avx2, float>;
template struct bench_gflops<avx512, float>;
template struct bench_gflops<avx2_plus, float>;

} // namespace x86

#    endif

} // namespace dabun

#endif
