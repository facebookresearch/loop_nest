// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/core.hpp"
#include <type_traits>

namespace dabun
{

template <typename T>
constexpr inline T ceil_div(T a, identity_type_t<T> b) noexcept
{
    return (a + b - 1) / b;
}

template <typename T>
constexpr inline T round_up(T a, identity_type_t<T> b) noexcept
{
    return ceil_div(a, b) * b;
}

// Equals to the number of iterations of the loop
// for (T i = from; i < to; i += stride)
// Assumes from <= to
template <typename T>
constexpr inline auto num_iterations(T from, identity_type_t<T> to,
                                     identity_type_t<T> stride) noexcept
    -> std::enable_if_t<std::is_integral_v<T>, T>
{
    return ceil_div(to - from, stride);
}

} // namespace dabun
