// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/core.hpp"
#include "dabun/math.hpp"
#include "dabun/thread/cpu_pool.hpp"

#include <atomic>
#include <functional>
#include <type_traits>
#include <vector>

namespace dabun::thread
{

template <class Int, class Fn>
inline auto naive_parallel_for(cpu_pool& working_cpu_pool, Int from,
                               identity_type_t<Int> to,
                               identity_type_t<Int> stride, Fn&& fn)
    -> std::enable_if_t<
        std::is_invocable_v<std::decay_t<Fn>, Int> ||
        std::is_invocable_v<std::decay_t<Fn>, cpu_context const&, Int>>
{
    std::vector<std::function<void(cpu_context const&)>> tasks(
        num_iterations(from, to, stride));

    for (auto i = from; i < to; i += stride)
    {
        tasks[i] = [&fn, i]([[maybe_unused]] cpu_context const& ctx)
        {
            if constexpr (std::is_invocable_v<std::decay_t<Fn>, Int>)
            {
                fn(i);
            }
            else
            {
                fn(ctx, i);
            }
        };
    }

    working_cpu_pool.execute(tasks);
}

template <class Int, class Fn>
inline auto single_queue_parallel_for(cpu_pool& working_cpu_pool, Int from,
                                      identity_type_t<Int> to,
                                      identity_type_t<Int> stride, Fn&& fn)
    -> std::enable_if_t<
        std::is_invocable_v<std::decay_t<Fn>, Int> ||
        std::is_invocable_v<std::decay_t<Fn>, cpu_context const&, Int>>
{
    alignas(hardware_destructive_interference_size) std::atomic<Int> i{from};

    alignas(hardware_destructive_interference_size) auto task =
        [&i, &fn, &stride, &to]([[maybe_unused]] cpu_context const& ctx)
    {
        for (Int idx = i.fetch_add(stride, std::memory_order_relaxed); idx < to;
             idx     = i.fetch_add(stride, std::memory_order_relaxed))
        {
            if constexpr (std::is_invocable_v<std::decay_t<Fn>, Int>)
            {
                fn(idx);
            }
            else
            {
                fn(ctx, idx);
            }
        }
    };

    working_cpu_pool.execute_on_all_cpus(task);
}

} // namespace dabun::thread
