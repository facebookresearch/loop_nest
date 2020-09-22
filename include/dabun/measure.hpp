#pragma once

#include <algorithm>
#include <chrono>
#include <limits>

namespace dabun
{

template <class Fn>
double measure_fastest(Fn&& fn, int iterations = 1)
{
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = start;

    double ret = std::numeric_limits<double>::max();

    for (int i = 0; i < iterations; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        fn();
        end = std::chrono::high_resolution_clock::now();

        auto new_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();

        ret = std::min(ret, static_cast<double>(new_time) / 1e9);
    }

    return ret;
}

} // namespace dabun
