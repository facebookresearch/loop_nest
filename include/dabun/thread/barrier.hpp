// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/core.hpp"
#include "dabun/isa.hpp"
#include "dabun/thread/core.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <new>

namespace dabun::thread
{

class alignas(hardware_destructive_interference_size) spinning_barrier
{
private:
    alignas(hardware_constructive_interference_size) std::size_t const
        barrier_threshold;

    alignas(hardware_constructive_interference_size)
        std::atomic<std::size_t> num_arrived{0};

    alignas(hardware_constructive_interference_size)
        std::atomic<std::size_t> generation{0};

    hardware_constructive_interference_padding<std::atomic<std::size_t>> pad;

public:
    explicit spinning_barrier(std::size_t threshold)
        : barrier_threshold(threshold)
    {
        strong_assert(threshold > 0);
    }

    bool arrive_and_wait()
    {
        auto generation_at_arrival = generation.load(std::memory_order_relaxed);

        if (num_arrived.fetch_add(static_cast<std::size_t>(1)) ==
            barrier_threshold - 1)
        {
            // Last arrival
            num_arrived = 0;

            generation.store(generation_at_arrival + 1,
                             std::memory_order_release);

            return true;
        }

        while (generation.load(std::memory_order_relaxed) ==
               generation_at_arrival)
        {
            DABUN_THREAD_CPU_RELAX();
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        return false;
    }
};

class alignas(hardware_destructive_interference_size) default_barrier
{
private:
    std::mutex              mutex_{};
    std::condition_variable cv_{};
    std::size_t             num_arrived = 0;
    std::size_t             generation  = 0;
    std::size_t const       barrier_threshold;

public:
    std::size_t arrival_count() const { return num_arrived; }

    explicit default_barrier(std::size_t threshold)
        : barrier_threshold(threshold)
    {
        strong_assert(threshold > 0);
    }

    bool arrive_and_wait()
    {
        std::unique_lock lock(mutex_);

        auto generation_at_arrival = generation;

        if (++num_arrived == barrier_threshold)
        {
            ++generation;
            num_arrived = 0;
            lock.unlock();
            cv_.notify_all();

            return true;
        }

        while (generation == generation_at_arrival)
        {
            cv_.wait(lock);
        }

        return false;
    }
};

} // namespace dabun::thread
