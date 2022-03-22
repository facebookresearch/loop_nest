// Copyright 2004-2021 Facebook Inc, 2021-present Meta Inc; All Rights Reserved.

#pragma once

#include <iostream>

#include "dabun/aligned_wrapper.hpp"
#include "dabun/thread/barrier.hpp"
#include "dabun/thread/core.hpp"
#include "dabun/thread/cpu_set.hpp"

#include <sched.h>

#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
#include <vector>

namespace dabun::thread
{

struct cpu_context
{
    std::size_t cpu_index;
};

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

class alignas(hardware_destructive_interference_size) cpu_pool
{
private:
    static constexpr std::size_t all_execute_the_same =
        std::numeric_limits<std::size_t>::max();

private:
    dabun::detail::primitive_aligned_wrapper<
        std::size_t, hardware_destructive_interference_size> const size_;
    cpu_set               original_cpu_set_;
    bool                  restore_original_cpu_set_ = false;
    thread_owner_enforcer enforcer_;
    cpu_context           zeroth_cpu_context_{0};

    alignas(hardware_destructive_interference_size)
        spinning_barrier spinning_barrier_;

    alignas(hardware_destructive_interference_size)
        default_barrier sleeping_barrier_;

    alignas(hardware_destructive_interference_size)
        std::function<void(cpu_context const&)> const* tasks_ = nullptr;
    std::size_t tasks_size_                                   = 0;

    alignas(hardware_destructive_interference_size)
        std::function<void(cpu_context const&)> const sleep_function_;

    std::vector<std::function<void(cpu_context const&)>> const
        sleep_function_tasks_;

    alignas(hardware_destructive_interference_size) bool is_sleeping_ = false;

    void cpu_working_loop(std::size_t idx, std::optional<int> cpu_id)
    {
        if (cpu_id) // Has to bind to a particular core
        {
            bind_to_core(*cpu_id);
        }

        cpu_context working_cpu_context = {idx};

        // Signal to the constructor that we are done initializing
        spinning_barrier_.arrive_and_wait();

        do
        {
            // Wait for a kernel;
            spinning_barrier_.arrive_and_wait();

            // Special case indicating that we need to exit the loop
            if (tasks_ == nullptr)
            {
                // We don't really need to reset the affinity, as the
                // thread will be exiting.  We just indicate the
                // completion of the loop.
                spinning_barrier_.arrive_and_wait();
                return;
            }
            else
            {
                if (tasks_size_ == all_execute_the_same)
                {
                    tasks_[0](working_cpu_context);
                }
                else
                {
                    for (std::size_t i = idx; i < tasks_size_; i += size_)
                    {
                        tasks_[i](working_cpu_context);
                    }
                }
            }

            //     if (tasks_[idx]) // Non-empty task
            // {
            //     tasks_[idx]();
            // }

            {
                spinning_barrier_.arrive_and_wait();
            }

        } while (true);
    }

public:
    std::size_t size() const noexcept { return size_; }

private:
    void initialize_workers(std::vector<int> const* cpu_ids_ptr)
    {
        // Bind the main thread
        if (cpu_ids_ptr != nullptr)
        {
            restore_original_cpu_set_ = true;
            get_affinity(original_cpu_set_);
            bind_to_core(cpu_ids_ptr->operator[](0));
        }

        // Makes no sense to have a operating set with less than 2 workers.
        // TODO(zi) Special pathway for operating set with only one worker.
        strong_assert(size() > 1);

        for (std::size_t idx = 1; idx < size(); ++idx)
        {
            if (cpu_ids_ptr != nullptr)
            {
                std::thread worker(&cpu_pool::cpu_working_loop, this, idx,
                                   cpu_ids_ptr->operator[](idx));
                worker.detach();
            }
            else
            {
                std::thread worker(&cpu_pool::cpu_working_loop, this, idx,
                                   std::nullopt);
                worker.detach();
            }
        }

        // Wait for all workers to signal being initialized
        spinning_barrier_.arrive_and_wait();
    }

    struct basic_constructor_tag
    {
    };

    cpu_pool(basic_constructor_tag, std::size_t s)
        : size_(s)
        , spinning_barrier_(size_)
        , sleeping_barrier_(size_)
        , tasks_(nullptr)
        , sleep_function_([this](cpu_context const&)
                          { this->sleeping_barrier_.arrive_and_wait(); })
        , sleep_function_tasks_(size_, sleep_function_)
    {
    }

public:
    explicit cpu_pool(std::vector<int> const& cpu_ids)
        : cpu_pool(basic_constructor_tag{}, cpu_ids.size())
    {
        initialize_workers(&cpu_ids);
    }

    explicit cpu_pool(std::size_t num_cpus, bool bind_to_cores = false)
        : cpu_pool(basic_constructor_tag{}, num_cpus)
    {
        if (bind_to_cores)
        {
            std::vector<int> cpu_ids(num_cpus);
            std::iota(std::begin(cpu_ids), std::end(cpu_ids),
                      static_cast<std::size_t>(0));
            initialize_workers(&cpu_ids);
        }
        else
        {
            initialize_workers(nullptr);
        }
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
            assert(tasks_ == sleep_function_tasks_.data());
            assert(tasks_size_ == size_);

            // All other threads doing the task of waiting will be
            // done with the waiting task.
            sleeping_barrier_.arrive_and_wait();

            // This has to be set back to nullptr after all the other
            // threads have finished the task (the magic task of
            // waiting on the sleeping thread).
            tasks_      = nullptr;
            tasks_size_ = 0;

            // Sync after being done with the tasks
            // TODO(zi) Optimize this - we don't need the spinning
            // barrier sync in this particular pathway.
            spinning_barrier_.arrive_and_wait();

            is_sleeping_ = false;
            return true;
        }
        else // Needs to go into sleeping mode
        {
            assert(tasks_ == nullptr);
            assert(tasks_size_ == 0);

            tasks_      = sleep_function_tasks_.data();
            tasks_size_ = size_;

            // All other threads started the task [and the task is to
            // wait on the sleeping barrier :)_
            spinning_barrier_.arrive_and_wait();

            is_sleeping_ = true;
            return false;
        }
    }

    void to_sleeping_mode() { set_sleeping_mode(true); }

    void to_spinning_mode() { set_sleeping_mode(false); }

    ~cpu_pool()
    {
        enforcer_.enforce();

        assert(tasks_ == nullptr);
        assert(tasks_size_ == 0);

        set_sleeping_mode(false);
        spinning_barrier_.arrive_and_wait();

        // Restore main threads CPU set
        if (restore_original_cpu_set_)
        {
            set_affinity(original_cpu_set_);
        }
    }

    bool in_spinning_mode() const { return !is_sleeping_; }

    bool in_sleeping_mode() const { return !is_sleeping_; }

    void execute(std::function<void(cpu_context const&)> const* tasks,
                 std::size_t const                              tasks_size)
    {
        enforcer_.enforce();

        bool was_sleeping = set_sleeping_mode(false);

        assert(tasks_ == nullptr);
        assert(tasks_size_ == 0);

        tasks_      = tasks;
        tasks_size_ = tasks_size;

        spinning_barrier_.arrive_and_wait();

        for (std::size_t i = 0; i < tasks_size_; i += size_)
        {
            tasks[i](zeroth_cpu_context_);
        }

        spinning_barrier_.arrive_and_wait();

        tasks_      = nullptr;
        tasks_size_ = 0;

        set_sleeping_mode(was_sleeping);
    }

    void
    execute_on_all_cpus(std::function<void(cpu_context const&)> const& task)
    {
        enforcer_.enforce();

        bool was_sleeping = set_sleeping_mode(false);

        assert(tasks_ == nullptr);
        assert(tasks_size_ == 0);

        tasks_      = std::addressof(task);
        tasks_size_ = all_execute_the_same;

        spinning_barrier_.arrive_and_wait();

        task(zeroth_cpu_context_);

        spinning_barrier_.arrive_and_wait();

        tasks_      = nullptr;
        tasks_size_ = 0;

        set_sleeping_mode(was_sleeping);
    }

    // template <class Fn>
    // auto execute_on_all_cpus(Fn&& task) -> std::enable_if<
    //     !std::is_same_v<std::function<void(cpu_context const&)>,
    //     std::decay_t<Fn>> && std::is_convertible_v<std::decay_t<Fn>,
    //     std::function<void(cpu_context const&)>>>
    // {
    //     std::function<void(cpu_context const&)> fn = std::forward<Fn>(task);
    //     execute_on_all_cpus(fn);

    //     return {};
    // }

    void execute(std::function<void(cpu_context const&)> const* tasks)
    {
        execute(tasks, size_);
    }

    void
    execute(std::vector<std::function<void(cpu_context const&)>> const& tasks)
    {
        execute(tasks.data(), tasks.size());
    }
};

} // namespace dabun::thread
