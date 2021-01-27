#pragma once

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

#if defined(__aarch64__)

using fp16 = _Float16;

template <>
struct printable_fp<fp16>
{
    using type = float;
};

#endif

template <class Float>
using printable_fp_t = typename printable_fp<Float>::type;

template <class Float>
inline printable_fp_t<Float> printable(Float f)
{
    return static_cast<printable_fp_t<Float>>(f);
}

} // namespace dabun
