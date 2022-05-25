// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

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
