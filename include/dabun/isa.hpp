#pragma once

#ifndef DABUN_ISA

#if defined(__AVX512F__)
#define DABUN_ISA avx512
#elif defined(__aarch64__)
#define DABUN_ISA aarch64
#else // default to avx2
// #elif defined(__AVX2__)
#define DABUN_ISA avx2
// #error "ISA not supported"
#endif

#endif

namespace dabun
{

struct avx2
{
};
struct avx512
{
};
struct avx2_plus
{
};
struct aarch64
{
};

template <class>
struct isa_traits;

template <>
struct isa_traits<avx2>
{
    static constexpr int total_vector_registers = 16;
    static constexpr int vector_register_mask   = 1;
    static constexpr int vector_size            = 8;
};

template <>
struct isa_traits<avx512>
{
    static constexpr int total_vector_registers = 32;
    static constexpr int vector_register_mask   = 0;
    static constexpr int vector_size            = 16;
};

template <>
struct isa_traits<avx2_plus>
{
    static constexpr int total_vector_registers = 16;
    static constexpr int vector_register_mask   = 0;
    static constexpr int vector_size            = 8;
};

template <>
struct isa_traits<aarch64>
{
    static constexpr int total_vector_registers = 32;
    static constexpr int vector_register_mask   = 0;
    static constexpr int vector_size            = 4;
};

} // namespace dabun
