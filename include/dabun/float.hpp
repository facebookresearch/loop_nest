// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#if !defined(__aarch64__) ||                                                   \
    !defined(__APPLE__) // Should also somehow detect whether _Float16 is
                        // available
#    include "dabun/third_party/half.hpp"

#endif

#include <type_traits>

namespace dabun
{

using fp32 = float;

template <class>
struct printable_fp;

template <>
struct printable_fp<float>
{
    using type = float;
};

template <>
struct printable_fp<double>
{
    using type = float;
};

template <class>
struct is_fp16_t : std::false_type
{
};

#if defined(__aarch64__) &&                                                    \
    defined(__APPLE__) // Should also somehow detect whether _Float16 is
                       // available

using fp16 = _Float16;

#else

using fp16 = half_float::half;

#endif

template <>
struct printable_fp<fp16>
{
    using type = float;
};

template <>
struct is_fp16_t<fp16> : std::true_type
{
};

template <class T>
inline constexpr bool is_fp16_v = is_fp16_t<T>::value;

template <class Float>
using printable_fp_t = typename printable_fp<Float>::type;

template <class Float>
inline printable_fp_t<Float> printable(Float f)
{
    return static_cast<printable_fp_t<Float>>(f);
}

// inline bool operator<(fp16 const &lhs, fp16 const &rhs)
// {
//     return static_cast<float>(lhs) < static_cast<float>(rhs);
// }

} // namespace dabun
