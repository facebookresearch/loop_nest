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
    auto end = std::chrono::high_resolution_clock::now();
    auto nsecs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    for (int i = 1; i < iterations; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        fn();
        end = std::chrono::high_resolution_clock::now();

        auto new_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();

        nsecs = std::min(nsecs, new_time);
    }

    return static_cast<double>(nsecs) / 1e9;
}

} // namespace dabun
