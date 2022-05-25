// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <boost/predef.h>

#if !defined(__APPLE__)

#    include "dabun/core.hpp"
#    include <memory>
#    include <sched.h>

namespace dabun::thread
{

class cpu_set
{
public:
    using native_handle_type = cpu_set_t;

private:
    std::unique_ptr<native_handle_type> handle;

public:
    cpu_set()
        : handle(std::make_unique<native_handle_type>())
    {
        CPU_ZERO(handle.get());
    }

    cpu_set& operator=(cpu_set const& other)
    {
        if (this != &other)
        {
            CPU_ZERO(handle.get());
            CPU_OR(handle.get(), handle.get(), other.handle.get());
        }
        return *this;
    }

    cpu_set(cpu_set const& other) { *this = other; }

    cpu_set& operator=(cpu_set&&) = default;
    cpu_set(cpu_set&&)            = default;

    native_handle_type&       native_handle() { return *handle; }
    native_handle_type const& native_handle() const { return *handle; }

public:
    void clear_all() { CPU_ZERO(handle.get()); }

    void zero() { CPU_ZERO(handle.get()); }

    void set(int cpu) { CPU_SET(cpu, handle.get()); }

    void clr(int cpu) { CPU_CLR(cpu, handle.get()); }

    bool is_set(int cpu) const { return CPU_ISSET(cpu, handle.get()); }

    int count() const { return CPU_COUNT(handle.get()); }

    friend bool operator==(cpu_set const& lhs, cpu_set const& rhs)
    {
        return CPU_EQUAL(lhs.handle.get(), rhs.handle.get());
    }
};

inline void get_affinity(cpu_set& set)
{
    strong_assert(sched_getaffinity(0, sizeof(cpu_set::native_handle_type),
                                    std::addressof(set.native_handle())) == 0);
}

inline void set_affinity(cpu_set const& set)
{
    strong_assert(sched_setaffinity(0, sizeof(cpu_set::native_handle_type),
                                    std::addressof(set.native_handle())) == 0);
}

inline void bind_to_core(int core)
{
    cpu_set s;
    s.set(core);
    set_affinity(s);
}

} // namespace dabun::thread

#else

#    include "dabun/core.hpp"
#    include <bitset>
#    include <memory>
#    include <thread>

namespace dabun::thread
{

class cpu_set
{
public:
    using native_handle_type = std::bitset<1024>;

private:
    std::unique_ptr<native_handle_type> handle;

public:
    cpu_set()
        : handle(std::make_unique<native_handle_type>())
    {
    }

    cpu_set& operator=(cpu_set const& other)
    {
        if (this != &other)
        {
            *handle = *other.handle;
        }
        return *this;
    }

    cpu_set(cpu_set const& other) { *this = other; }

    cpu_set& operator=(cpu_set&&) = default;
    cpu_set(cpu_set&&)            = default;

    native_handle_type&       native_handle() { return *handle; }
    native_handle_type const& native_handle() const { return *handle; }

public:
    void clear_all() { handle->reset(); }

    void zero() { handle->reset(); }

    void set(int cpu) { handle->set(cpu, true); }

    void clr(int cpu) { handle->set(cpu, false); }

    bool is_set(int cpu) const { return handle->test(cpu); }

    int count() const { return handle->count(); }

    friend bool operator==(cpu_set const& lhs, cpu_set const& rhs)
    {
        return *lhs.handle == *rhs.handle;
    }
};

inline void get_affinity(cpu_set& set)
{
    set.zero();
    for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i)
    {
        set.set(i);
    }
}

inline void set_affinity(cpu_set const&) {}

inline void bind_to_core(int) {}

} // namespace dabun::thread

#endif
