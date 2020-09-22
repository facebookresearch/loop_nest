#pragma once

namespace dabun
{

template <typename T>
T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
T round_up(T a, T b)
{
    return ceil_div(a, b) * b;
}

} // namespace dabun
