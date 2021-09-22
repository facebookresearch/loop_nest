#pragma once

#include <type_traits>

namespace dabun
{

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

#if defined(__aarch64__) // Should also somehow detect whether _Float16 is
                         // available

using fp16 = _Float16;

template <>
struct printable_fp<fp16>
{
    using type = float;
};

template <>
struct is_fp16_t<fp16> : std::true_type
{
};

#endif

template <class T>
inline constexpr bool is_fp16_v = is_fp16_t<T>::value;

template <class Float>
using printable_fp_t = typename printable_fp<Float>::type;

template <class Float>
inline printable_fp_t<Float> printable(Float f)
{
    return static_cast<printable_fp_t<Float>>(f);
}

} // namespace dabun
