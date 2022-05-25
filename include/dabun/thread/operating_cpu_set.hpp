// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/thread/cpu_set.hpp"

#include <iostream>

#include "dabun/thread/barrier.hpp"
#include "dabun/thread/core.hpp"

#include <sched.h>

#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace dabun::thread
{

class thread_owner_enforcer
{
private:
    std::thread::id thread_id_;

public:
    thread_owner_enforcer()
        : thread_id_(std::this_thread::get_id())
    {
    }

    void enforce() { strong_assert(thread_id_ == std::this_thread::get_id()); }
};

class alignas(hardware_destructive_interference_size) operating_cpu_set
{
private:
    alignas(hardware_destructive_interference_size) std::size_t const size_;
    cpu_set               original_cpu_set_;
    thread_owner_enforcer enforcer_;

    alignas(hardware_destructive_interference_size)
        spinning_barrier spinning_barrier_;

    alignas(hardware_destructive_interference_size)
        default_barrier sleeping_barrier_;

    alignas(hardware_destructive_interference_size)
        std::function<void()> const* kernels_ = nullptr;

    alignas(hardware_destructive_interference_size) std::function<void()> const
        sleep_function_;

    std::vector<std::function<void()>> const sleep_function_kernels_;

    alignas(hardware_destructive_interference_size) bool is_sleeping_ = false;
    // alignas(hardware_destructive_interference_size) cpu_set_t old_set_;

    void operating_cpu_loop(std::size_t idx, std::optional<int> core_id)
    {
        if (core_id) // Has to bind to a particular core
        {
            bind_to_core(*core_id);
        }

        // Signal to the constructor that we are done initializing
        spinning_barrier_.arrive_and_wait();

        do
        {
            // Wait for a kernel;
            spinning_barrier_.arrive_and_wait();

            // Special case indicating that we need to exit the loop
            if (kernels_ == nullptr)
            {
                // We don't really need to reset the affinity, as the
                // thread will be exiting.  We just indicate the
                // completion of the loop.
                spinning_barrier_.arrive_and_wait();
                return;
            }
            else if (kernels_[idx]) // Non-empty task
            {
                kernels_[idx]();
            }

            {
                spinning_barrier_.arrive_and_wait();
            }

        } while (true);
    }

public:
    std::size_t size() const noexcept { return size_; }

public:
    explicit operating_cpu_set(std::vector<int> const& core_ids)
        : size_(core_ids.size())
        , spinning_barrier_(size_)
        , sleeping_barrier_(size_)
        , kernels_(nullptr)
        , sleep_function_([this]()
                          { this->sleeping_barrier_.arrive_and_wait(); })
        , sleep_function_kernels_(size_, sleep_function_)
    // , old_set_()

    {

        // Bind the main thread
        {
            get_affinity(original_cpu_set_);
            bind_to_core(core_ids[0]);
        }

        // Makes no sense to have a operating set with less than 2 workers.
        // TODO(zi) Special pathway for operating set with only one worker.
        strong_assert(size() > 1);

        for (std::size_t idx = 1; idx < size(); ++idx)
        {
            std::thread worker(&operating_cpu_set::operating_cpu_loop, this,
                               idx, core_ids[idx]);

            worker.detach(); // Not really necessary, but provides clarity to
                             // the code reader.
        }

        // Wait for all workers to signal being initialized
        spinning_barrier_.arrive_and_wait();
    }

    bool set_sleeping_mode(bool sleep_mode)
    {
        enforcer_.enforce();

        if (sleep_mode == is_sleeping_)
        {
            return sleep_mode;
        }

        if (is_sleeping_) // Needs to go into spinning mode, which is
                          // just signaling the sleeping thread on
                          // which all other threads are waiting.
        {
            assert(kernels_ == sleep_function_kernels_.data());

            // All other threads doing the task of waiting will be
            // done with the waiting task.
            sleeping_barrier_.arrive_and_wait();

            // This has to be set back to nullptr after all the other
            // threads have finished the task (the magic task of
            // waiting on the sleeping thread).
            kernels_ = nullptr;

            // Sync after being done with the tasks
            // TODO(zi) Optimize this - we don't need the spinning
            // barrier sync in this particular pathway.
            spinning_barrier_.arrive_and_wait();

            is_sleeping_ = false;
            return true;
        }
        else // Needs to go into sleeping mode
        {
            assert(kernels_ == nullptr);
            kernels_ = sleep_function_kernels_.data();

            // All other threads started the task [and the task is to
            // wait on the sleeping barrier :)_
            spinning_barrier_.arrive_and_wait();

            is_sleeping_ = true;
            return false;
        }
    }

    void to_sleeping_mode() { set_sleeping_mode(true); }

    void to_spinning_mode() { set_sleeping_mode(false); }

    ~operating_cpu_set()
    {
        enforcer_.enforce();

        assert(kernels_ == nullptr);
        set_sleeping_mode(false);
        spinning_barrier_.arrive_and_wait();

        // Restore main threads CPU set
        {
            set_affinity(original_cpu_set_);
        }
    }

    bool in_spinning_mode() const { return !is_sleeping_; }

    bool in_sleeping_mode() const { return !is_sleeping_; }

    void execute(std::function<void()> const* kernels)
    {
        enforcer_.enforce();

        bool was_sleeping = set_sleeping_mode(false);

        assert(kernels_ == nullptr);
        kernels_ = kernels;
        assert(kernels_ != nullptr);

        spinning_barrier_.arrive_and_wait();

        if (kernels[0])
        {
            kernels[0]();
        }

        spinning_barrier_.arrive_and_wait();

        set_sleeping_mode(was_sleeping);
    }

    void execute(std::vector<std::function<void()>> const& kernels)
    {
        assert(kernels.size() == size());
        execute(kernels.data());
    }
};

} // namespace dabun::thread
