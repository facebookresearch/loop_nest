#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "AlignedVec.h"

#define LOOP_NEST_STRINGIFY_0(s) #s
#define LOOP_NEST_STRINGIFY(s) LOOP_NEST_STRINGIFY_0(s)

#define strong_assert(condition)                                               \
    if (!(condition))                                                          \
    {                                                                          \
        throw std::runtime_error(LOOP_NEST_STRINGIFY(                          \
            condition) " failed file: " __FILE__                               \
                       " line: " LOOP_NEST_STRINGIFY((__LINE__)));             \
    }                                                                          \
    static_cast<void>(0)

// FROM: https://en.cppreference.com/w/cpp/utility/variant/visit

// helper type for the visitor #4
template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

template <class Float>
void apply_relu(Float* Begin, float* End)
{
    for (; Begin != End; ++Begin)
    {
        *Begin = std::max(static_cast<Float>(0), *Begin);
    }
}

template <class Float>
Float maxAbsDiff(Float const* LBegin, Float const* LEnd, Float const* RBegin)
{
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        res = std::max(res, std::abs(*LBegin - *RBegin));
    }
    return res;
}

template <class Float>
Float maxAbsDiffVerbose(Float const* LBegin, Float const* LEnd,
                        Float const* RBegin)
{
    int   off = 0;
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        std::cout << off++ << " : " << (*LBegin) << " " << (*RBegin) << " "
                  << std::abs(*LBegin - *RBegin) << "\n";
        res = std::max(res, std::abs(*LBegin - *RBegin));
    }
    return res;
}

template <class Float>
Float maxAbsDiffVerbose(Float const* LBegin, Float const* LEnd,
                        Float const* RBegin, float delta)
{
    int   off = 0;
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        if (std::abs(*LBegin - *RBegin) > delta)
        {
            std::cout << off << " : " << (*LBegin) << " " << (*RBegin) << " "
                      << std::abs(*LBegin - *RBegin) << "\n";
        }
        res = std::max(res, std::abs(*LBegin - *RBegin));
        off++;
    }
    return res;
}

template <class Float>
aligned_vector<Float> getRandomVector(unsigned size,
                                      unsigned extra_elements = 16)
{
    aligned_vector<Float> res(size + extra_elements);

    std::random_device rd;
    std::mt19937       gen(0); //rd());

    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (auto& f : res)
    {
        f = dis(gen);
    }

    return res;
}

template <class Float>
aligned_vector<Float> getZeroVector(unsigned size, unsigned extra_elements = 16)
{
    aligned_vector<Float> res(size + extra_elements);
    return res;
}

template <class Fn>
double measureFastestWithWarmup(Fn&& fn, int warmupIterations,
                                int measuredIterations = 1)
{
    for (int i = 0; i < warmupIterations; ++i)
    {
        fn();
    }

    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    auto nsecs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    for (int i = 1; i < measuredIterations; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        fn();
        end = std::chrono::high_resolution_clock::now();

        auto new_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();

        // LN_LOG(INFO) << "T: " << new_time << "\n";
        nsecs = std::min(nsecs, new_time);
    }

    return static_cast<double>(nsecs) / 1e9;
}

inline std::uint64_t rdtsc()
{
#if !defined(LOOP_NEST_ARM)
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((std::uint64_t)lo) | (((std::uint64_t)hi) << 32);
#else
    return 0;
#endif
}

template <class Fn>
double measureMinCyclesWithWarmup(Fn&& fn, int warmupIterations,
                                  int measuredIterations = 1)
{
    for (int i = 0; i < warmupIterations; ++i)
    {
        fn();
    }

    auto start = rdtsc();
    fn();
    auto end = rdtsc();
    auto cyc = end - start;

    for (int i = 1; i < measuredIterations; ++i)
    {
        start = rdtsc();
        fn();
        end = rdtsc();

        auto new_time = end - start;
        // LN_LOG(INFO) << "T: " << new_time << "\n";
        cyc = std::min(cyc, new_time);
    }

    return static_cast<double>(cyc);
}

template <class BaseLineImpl, class JITImpl>
void check_correctness(BaseLineImpl&& baseline_fn, JITImpl&& jit_fn, int A_size,
                       int B_size, int C_size, int alpha = 0)
{
    auto A = getRandomVector<float>(A_size);
    auto B = getRandomVector<float>(B_size);

    auto CN = aligned_vector<float>(C_size);
    auto CJ = std::vector<float>(C_size);

    baseline_fn(CN.data(), A.data(), B.data());
    jit_fn(CJ.data(), A.data(), B.data(), alpha);

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(CJ.data(), CJ.data() + C_size, CN.data()) << "\n";
}

template <class Fn>
void bench_implementation(Fn&& fn, int A_size, int B_size, int C_size,
                          double gflops, int warmup = 5, int iters = 10)
{
    auto A = getRandomVector<float>(A_size);
    auto B = getRandomVector<float>(B_size);
    auto C = std::vector<float>(C_size);

    auto secs = measureFastestWithWarmup(
        [&]() { fn(C.data(), A.data(), B.data(), 0); }, warmup, iters);

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
}

template <class Fn>
void bench_implementation_fmas_per_cycle(Fn&& fn, int A_size, int B_size,
                                         int C_size, double flops,
                                         int warmup = 5, int iters = 10)
{
    auto A = getRandomVector<float>(A_size);
    auto B = getRandomVector<float>(B_size);
    auto C = std::vector<float>(C_size);

    auto secs = measureMinCyclesWithWarmup(
        [&]() { fn(C.data(), A.data(), B.data(), 0); }, warmup, iters);

    std::cout << "FLOPS per CYCLE: " << (flops / secs) << "\n";
}
