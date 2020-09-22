// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/namespace.hpp"

#if defined(DABUN_ARM)
#include "dabun/arm/peak_gflops.hpp"
#else
#include "dabun/x86/peak_gflops.hpp"
#endif

namespace dabun
{

template <class T>
double peak_gflops(T const&, int iterations = 1000000)
{
    return DABUN_ISA_NAMESPACE ::bench_gflops<T>::do_bench(iterations);
}

} // namespace dabun
