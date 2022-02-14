// Copyright 2004-2021 Facebook Inc, 2021-present Meta Inc; All Rights Reserved.

#pragma once

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

class alignas(hardware_destructive_interference_size) operating_cpu_set
{
private:
    alignas(hardware_destructive_interference_size)
        spinning_barrier spinning_barrier_;

    alignas(hardware_destructive_interference_size)
        default_barrier sleeping_barrier_;

    alignas(hardware_destructive_interference_size) std::size_t
        num_operating_cpus_;

    // alignas(hardware_destructive_interference_size) cpu_set_t old_set_;

    alignas(hardware_destructive_interference_size)
        std::function<void()>* kernels_ = nullptr;

    alignas(hardware_destructive_interference_size)
        std::function<void()> sleep_function_;

    std::vector<std::function<void()>> sleep_function_kernels_;

    alignas(hardware_destructive_interference_size) bool is_sleeping_ = false;

    void operating_cpu_loop(std::size_t idx, std::optional<int> core_id)
    {
        if (core_id) // Has to bind to a particular core
        {
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
    explicit operating_cpu_set(std::vector<int> const& core_ids)
        : spinning_barrier_(core_ids.size())
        , sleeping_barrier_(core_ids.size())
        , num_operating_cpus_(core_ids.size())
        // , old_set_()

    {
        sleep_function_ = [this]()
        { this->sleeping_barrier_.arrive_and_wait(); };

        sleep_function_kernels_.reserve(core_ids.size());

        for (std::size_t i = 0; i < core_ids.size(); ++i)
        {
            sleep_function_kernels_.push_back(sleep_function_);
        }

        // Bind the main thread
        {
        }

        // Makes no sense to have a operating set with less than 2 workers.
        // TODO(zi) Special pathway for operating set with only one worker.
        strong_assert(core_ids.size() > 1);

        for (std::size_t idx = 1; idx < core_ids.size(); ++idx)
        {
            std::thread worker(&operating_cpu_set::operating_cpu_loop, this,
                               idx, core_ids[idx]);

            worker.detach(); // Not really necessary, but provides clarity to
                             // the code reader.
        }

        // Wait for all workers to signal being initialized
        spinning_barrier_.arrive_and_wait();

        // Restore main threads CPU set
        {
        }
    }

    bool set_sleeping_mode(bool sleep_mode)
    {
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
        assert(kernels_ == nullptr);
        set_sleeping_mode(false);
        spinning_barrier_.arrive_and_wait();
    }
};

} // namespace dabun::thread
