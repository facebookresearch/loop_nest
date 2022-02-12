// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/thread/barrier.hpp"
#include "dabun/thread/core.hpp"

#include <sched.h>
#include <functional>

namespace dabun::thread
{

class alignas(hardware_destructive_interference_size) operating_cpu_set
{
private:
    alignas(hardware_destructive_interference_size)
        spinning_barrier spinning_barrier_;
    alignas(hardware_destructive_interference_size) std::size_t
        num_operating_cpus_;
    alignas(hardware_destructive_interference_size) cpu_set_t old_set_;
    alignas(
        hardware_destructive_interference_size) std::function<void()> *kernels;
};

} // namespace dabun::thread
