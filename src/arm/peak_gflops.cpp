// Copyright 2004-present Facebook. All Rights Reserved.

#include "dabun/arm/peak_gflops.hpp"

#ifdef DABUN_NOT_HEADER_ONLY

namespace dabun::arm
{

template struct bench_gflops<aarch64, fp32>;
template struct bench_gflops<aarch64, fp16>;

} // namespace dabun::arm

#endif
