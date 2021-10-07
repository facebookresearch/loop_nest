// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"
#include "dabun/namespace.hpp"

#if defined(DABUN_ARCH_AARCH64)
#include "dabun/arm/peak_gflops.hpp"
#else
#include "dabun/x86/peak_gflops.hpp"
#endif

namespace dabun
{

template <class T, class A>
double peak_gflops(int iterations = 1000000)
{
    return DABUN_ISA_NAMESPACE ::bench_gflops<T, A>::do_bench(iterations);
}

} // namespace dabun
