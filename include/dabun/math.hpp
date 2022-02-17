// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/core.hpp"

namespace dabun
{

template <typename T>
constexpr T ceil_div(T a, identity_type_t<T> b)
{
    return (a + b - 1) / b;
}

template <typename T>
constexpr T round_up(T a, identity_type_t<T> b)
{
    return ceil_div(a, b) * b;
}

} // namespace dabun
