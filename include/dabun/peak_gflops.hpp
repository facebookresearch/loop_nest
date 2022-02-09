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
    auto measurement =
        DABUN_ISA_NAMESPACE ::bench_gflops<T, A>::do_bench(iterations);
    return measurement.first / measurement.second;
}

template <class T, class A>
double measure_peak_gflops(double secs, int max_iterations = 1000000)
{
    int cur_it = 1;
    auto measurement =
        DABUN_ISA_NAMESPACE ::bench_gflops<T, A>::do_bench(cur_it);

    while (measurement.first < secs && cur_it <= max_iterations)
    {
        cur_it *= 2;
        measurement =
            DABUN_ISA_NAMESPACE ::bench_gflops<T, A>::do_bench(cur_it);
    }

    return measurement.first / measurement.second;
}


} // namespace dabun
