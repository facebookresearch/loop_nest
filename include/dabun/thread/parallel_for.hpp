// Copyright 2004-2021 Facebook Inc, 2021-present Meta Inc; All Rights Reserved.

#pragma once

#include "dabun/core.hpp"
#include "dabun/math.hpp"
#include "dabun/thread/cpu_pool.hpp"

#include <functional>
#include <type_traits>
#include <vector>
#include <atomic>

namespace dabun::thread
{

template <class Int, class Fn>
inline auto naive_parallel_for(cpu_pool& working_cpu_pool, Int from,
                               identity_type_t<Int> to,
                               identity_type_t<Int> stride, Fn&& fn)
    -> std::enable_if_t<std::is_invocable_v<std::decay_t<Fn>, Int>>
{
    std::vector<std::function<void()>> tasks(num_iterations(from, to, stride));

    for (auto i = from; i < to; i += stride)
    {
        tasks[i] = [&fn, i]() { fn(i); };
    }

    working_cpu_pool.execute(tasks);
}

template <class Int, class Fn>
inline auto single_queue_parallel_for(cpu_pool& working_cpu_pool, Int from,
                                      identity_type_t<Int> to,
                                      identity_type_t<Int> stride, Fn&& fn)
    -> std::enable_if_t<std::is_invocable_v<std::decay_t<Fn>, Int>>
{
    alignas(hardware_destructive_interference_size)
        std::atomic<Int> i{from};

    alignas(hardware_destructive_interference_size)
        auto task = [&i, &fn, &stride, &to]()
        {
            for (Int idx = i.fetch_add(stride, std::memory_order_relaxed);
                 idx < to;
                 idx = i.fetch_add(stride, std::memory_order_relaxed))
            {
                fn(idx);
            }
        };

    working_cpu_pool.execute_on_all_cpus(task);
}

} // namespace dabun::thread
