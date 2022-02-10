// Copyright 2004-present Facebook. All Rights Reserved.

#include "dabun/x86/peak_gflops.hpp"

#ifdef DABUN_NOT_HEADER_ONLY

template struct dabun::x86::bench_gflops<dabun::avx2, float>;
template struct dabun::x86::bench_gflops<dabun::avx512, float>;
template struct dabun::x86::bench_gflops<dabun::avx2_plus, float>;

#endif
