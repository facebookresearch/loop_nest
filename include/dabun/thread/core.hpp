// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/core.hpp"
#include "dabun/isa.hpp"

#include <cstddef>

#if defined(DABUN_ARCH_AARCH64)
#    define DABUN_THREAD_CPU_RELAX() asm __volatile__("yield" ::: "memory")
#else
#    define DABUN_THREAD_CPU_RELAX() asm __volatile__("pause" ::: "memory")
#endif

namespace dabun::thread
{

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size  = 64;
#endif

template <class T>
struct hardware_constructive_interference_padding
{
    char padding[hardware_constructive_interference_size -
                 sizeof(T)]; // = {'\0'};
};

} // namespace dabun::thread
