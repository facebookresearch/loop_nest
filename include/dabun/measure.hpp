// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace dabun
{

struct time_duraton_measurement
{
    double shortest = std::numeric_limits<double>::max();
    double mean     = std::numeric_limits<double>::max();
    double median   = std::numeric_limits<double>::max();
};

struct flops_measurement
{
    double shortest = std::numeric_limits<double>::max();
    double mean     = std::numeric_limits<double>::max();
    double median   = std::numeric_limits<double>::max();
};

template <class Fn>
__attribute__((always_inline)) double measure_single_run(Fn&& fn)
{
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    fn();
    auto end = high_resolution_clock::now();

    auto nsecs = duration_cast<nanoseconds>(end - start).count();

    return static_cast<double>(nsecs) / 1e9;
}

template <class Fn>
double measure_fastest(Fn&& fn, int iterations = 1)
{
    using namespace std::chrono;

    auto nsecs = nanoseconds::max().count();

    for (int i = 0; i < iterations; ++i)
    {
        auto start = high_resolution_clock::now();
        fn();
        auto end = high_resolution_clock::now();

        auto new_time = duration_cast<nanoseconds>(end - start).count();

        nsecs = std::min(nsecs, new_time);
    }

    return static_cast<double>(nsecs) / 1e9;
}

template <class Fn>
double measure_fastest_time_limited(Fn&& fn, int iterations = 1,
                                    double seconds = 1.0)
{
    double ret        = std::numeric_limits<double>::max();
    double total_time = 0.0;

    for (int i = 0; i < iterations; ++i)
    {
        auto t = measure_single_run(fn);
        ret    = std::min(ret, t);
        total_time += t;
        if (total_time > seconds)
        {
            return ret;
        }
    }

    return ret;
}

template <class Fn>
double measure_mean(Fn&& fn, int iterations = 1, int warmup_iterations = 1)
{
    using namespace std::chrono;

    if (iterations <= 0)
    {
        return std::numeric_limits<double>::max();
    }

    for (int i = 0; i < warmup_iterations; ++i)
    {
        fn();
    }

    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        fn();
    }

    auto end   = high_resolution_clock::now();
    auto nsecs = duration_cast<nanoseconds>(end - start).count();

    return static_cast<double>(nsecs) / 1e9 / iterations;
}

template <class Fn>
double measure_median(Fn&& fn, int iterations = 1, int warmup_iterations = 1)
{
    using namespace std::chrono;

    std::vector<double> measurements(iterations);

    if (iterations <= 0)
    {
        return -1;
    }

    for (int i = 0; i < warmup_iterations; ++i)
    {
        fn();
    }

    for (int i = 0; i < iterations; ++i)
    {
        auto start = high_resolution_clock::now();
        fn();
        auto end        = high_resolution_clock::now();
        auto nsecs      = duration_cast<nanoseconds>(end - start).count();
        measurements[i] = static_cast<double>(nsecs) / 1e9;
    }

    std::sort(std::begin(measurements), std::end(measurements));

    return measurements[iterations / 2];
}

template <class Fn>
std::tuple<double, double, double> measure_all(Fn&& fn, int iterations = 1,
                                               int warmup_iterations = 1)
{
    using namespace std::chrono;

    auto fastest = nanoseconds::max().count();

    std::vector<double> measurements(iterations);

    if (iterations <= 0)
    {
        return {-1., -1., -1.};
    }

    for (int i = 0; i < warmup_iterations; ++i)
    {
        fn();
    }

    for (int i = 0; i < iterations; ++i)
    {
        auto start = high_resolution_clock::now();
        fn();
        auto end        = high_resolution_clock::now();
        auto nsecs      = duration_cast<nanoseconds>(end - start).count();
        measurements[i] = static_cast<double>(nsecs) / 1e9;

        fastest = std::min(fastest, nsecs);
    }

    std::sort(std::begin(measurements), std::end(measurements));

    double the_median = measurements[iterations / 2];
    double the_mean =
        std::accumulate(std::begin(measurements), std::end(measurements), 0.0) /
        measurements.size();
    double the_fastest = static_cast<double>(fastest) / 1e9;

    return {the_fastest, the_mean, the_median};
}

} // namespace dabun
