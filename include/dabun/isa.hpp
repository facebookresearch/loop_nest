// Copyright 2004-present Facebook. All Rights Reserved.

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
    static constexpr int fp16_vector_size       = 2;
};

} // namespace dabun

// Copyright 2004-present Facebook. All Rights Reserved.

// #pragma once

// For deprecated APIs
// #include "dabun/isa.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) ||           \
    defined(__x86_64)
#define DABUN_ARCH_X86_64
#elif defined(__aarch64__)
#define DABUN_ARCH_AARCH64
#else
#error "Unknown target architecture"
#endif

namespace dabun
{

enum class architecture_kind : int
{
    unknown = 0,
    x86_64  = 1,
    aarch64 = 2
};

enum class extension : int
{
    unknown = 0,

#if defined(DABUN_ARCH_X86_64)
    avx        = 1001,
    avx2       = 1002,
    avx512_ymm = 1003,
    avx512     = 1004
#elif defined(DABUN_ARCH_AARCH64)
    neon      = 2001,
    neon_fp16 = 2002
#endif
};

template <extension E>
struct extension_traits;

// TODO(zi) deprecate the following two
template <extension E>
struct extension_to_deprecated_ISA;

template <extension E>
using extension_to_deprecated_ISA_t =
    typename extension_to_deprecated_ISA<E>::type;

#if defined(DABUN_ARCH_X86_64)

template <>
struct extension_traits<extension::avx2>
{
    static constexpr architecture_kind architecture = architecture_kind::x86_64;
    static constexpr int               vector_register_bits = 256;
    static constexpr int               vector_size          = 32;
    static constexpr bool              has_mask_register    = false;
    static constexpr int               num_vector_registers = 32;
};

template <>
struct extension_traits<extension::avx512_ymm>
{
    static constexpr architecture_kind architecture = architecture_kind::x86_64;
    static constexpr int               vector_register_bits = 256;
    static constexpr int               vector_size          = 32;
    static constexpr bool              has_mask_register    = true;
    static constexpr int               num_vector_registers = 32;
};

template <>
struct extension_traits<extension::avx512>
{
    static constexpr architecture_kind architecture = architecture_kind::x86_64;
    static constexpr int               vector_register_bits = 512;
    static constexpr int               vector_size          = 64;
    static constexpr bool              has_mask_register    = false;
    static constexpr int               num_vector_registers = 32;
};

// TODO(zi) deprecate
template <>
struct extension_to_deprecated_ISA<extension::avx2>
{
    using type = avx2;
};
template <>
struct extension_to_deprecated_ISA<extension::avx512_ymm>
{
    using type = avx2_plus;
};
template <>
struct extension_to_deprecated_ISA<extension::avx512>
{
    using type = avx512;
};

#elif defined(DABUN_ARCH_AARCH64)

template <>
struct extension_traits<extension::neon>
{
    static constexpr architecture_kind architecture =
        architecture_kind::aarch64;
    static constexpr int  vector_register_bits = 128;
    static constexpr int  vector_size          = 16;
    static constexpr bool has_mask_register    = false;
    static constexpr int  num_vector_registers = 32;
};

template <>
struct extension_traits<extension::neon_fp16>
{
    static constexpr architecture_kind architecture =
        architecture_kind::aarch64;
    static constexpr int  vector_register_bits = 128;
    static constexpr int  vector_size          = 16;
    static constexpr bool has_mask_register    = false;
    static constexpr int  num_vector_registers = 32;
};

// TODO(zi) deprecate
template <>
struct extension_to_deprecated_ISA<extension::neon>
{
    using type = aarch64;
};
template <>
struct extension_to_deprecated_ISA<extension::neon_fp16>
{
    using type = aarch64;
};

#endif

} // namespace dabun
