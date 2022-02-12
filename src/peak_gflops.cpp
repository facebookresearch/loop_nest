// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef DABUN_HEADER_ONLY

#include "dabun/peak_gflops.hpp"

namespace dabun
{

#if defined(DABUN_ARCH_AARCH64)

namespace arm
{

template struct bench_gflops<aarch64, fp32>;
template struct bench_gflops<aarch64, fp16>;

} // namespace arm

#else

namespace x86
{

template struct bench_gflops<avx2, float>;
template struct bench_gflops<avx512, float>;
template struct bench_gflops<avx2_plus, float>;

} // namespace x86

#endif

} // namespace dabun

#endif
