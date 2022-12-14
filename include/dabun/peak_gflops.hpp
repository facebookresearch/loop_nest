// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"

#if defined(DABUN_ARCH_AARCH64)
#    include "dabun/arm/peak_gflops.hpp"
#else
#    include "dabun/x86/peak_gflops.hpp"
#endif

namespace dabun::impl
{

} // namespace dabun::impl

namespace dabun
{

namespace detail
{

template <class T, class A>
struct peak_gflops_impl
{
    static double peak_gflops(int iterations = 1000000);
    static double measure_peak_gflops(double secs,
                                      int    max_iterations = 1000000);
};

#if defined(DABUN_REQUIES_TEMPLATE_DEFINITION)

template <class T, class A>
double peak_gflops_impl<T, A>::peak_gflops(int iterations)
{
    auto measurement =
        DABUN_ISA_NAMESPACE ::bench_gflops<T, A>::do_bench(iterations);
    return measurement.first / measurement.second;
}

template <class T, class A>
double peak_gflops_impl<T, A>::measure_peak_gflops(double secs,
                                                   int    max_iterations)
{
    int  cur_it = 1;
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

#endif

#if defined(DABUN_MAYBE_EXTN_TPL_INSTNTON)

#    if defined(DABUN_ARCH_AARCH64)

DABUN_MAYBE_EXTN_TPL_INSTNTON struct peak_gflops_impl<aarch64, fp32_t>;
DABUN_MAYBE_EXTN_TPL_INSTNTON struct peak_gflops_impl<aarch64, fp16_t>;

#    else

DABUN_MAYBE_EXTN_TPL_INSTNTON struct peak_gflops_impl<avx2, float>;
DABUN_MAYBE_EXTN_TPL_INSTNTON struct peak_gflops_impl<avx512, float>;
DABUN_MAYBE_EXTN_TPL_INSTNTON struct peak_gflops_impl<avx2_plus, float>;

#    endif

#endif

} // namespace detail

template <class T, class A>
double peak_gflops(int iterations = 1000000)
{
    return detail::peak_gflops_impl<T, A>::peak_gflops(iterations);
}

template <class T, class A>
double measure_peak_gflops(double secs, int max_iterations = 1000000)
{
    return detail::peak_gflops_impl<T, A>::measure_peak_gflops(secs,
                                                               max_iterations);
}

} // namespace dabun
