// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

#define DABUN_STRINGIFY_0(s) #s
#define DABUN_STRINGIFY(s) DABUN_STRINGIFY_0(s)

#if defined(DABUN_REQUIES_TEMPLATE_DEFINITION) ||                              \
    defined(DABUN_MAYBE_EXTN_TPL_INSTNTON)
#    error The macros above cannot be defined at this stage!
#endif

#if defined(DABUN_HEADER_ONLY)
#    if defined(DABUN_COMPILING_LIBDABUN)
#        error Unsupported combination of defines
#    else
#        define DABUN_REQUIES_TEMPLATE_DEFINITION
#    endif
#else
#    if defined(DABUN_COMPILING_LIBDABUN)
#        define DABUN_REQUIES_TEMPLATE_DEFINITION
#        define DABUN_MAYBE_EXTN_TPL_INSTNTON template
#    else
#        define DABUN_MAYBE_EXTN_TPL_INSTNTON extern template
#    endif
#endif

#define strong_assert(condition)                                               \
    if (!(condition))                                                          \
    {                                                                          \
        throw std::runtime_error(                                              \
            DABUN_STRINGIFY(condition) " failed file: " __FILE__               \
                                       " line: " DABUN_STRINGIFY((__LINE__))); \
    }                                                                          \
    static_cast<void>(0)

namespace dabun
{

#ifndef NDEBUG
inline constexpr bool compiled_in_debug_mode = true;
#else
inline constexpr bool compiled_in_debug_mode = false;
#endif

// FROM: https://en.cppreference.com/w/cpp/utility/variant/visit

template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};

// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

template <class T>
struct identity_type
{
    using type = T;
};

template <class T>
using identity_type_t = typename identity_type<T>::type;

// Sourced from https://en.cppreference.com/w/cpp/numeric/bit_cast
// to enable bit_cast from C++20
template <class To, class From>
typename std::enable_if_t<sizeof(To) == sizeof(From) &&
                              std::is_trivially_copyable_v<From> &&
                              std::is_trivially_copyable_v<To>,
                          To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept
{
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation additionally requires destination type "
                  "to be trivially constructible");

    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

#define DABUN_OP_RESULT_TYPE(OP, T1, T2)                                       \
    decltype(std::declval<std::decay_t<T1>>()                                  \
                 OP std::declval<std::decay_t<T2>>())

#define DABUN_ALWAYS_INLINE __attribute__((always_inline)) inline

} // namespace dabun
