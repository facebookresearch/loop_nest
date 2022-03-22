// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

namespace dabun
{

template <class T, std::size_t N>
struct array : std::array<T, N>
{
};

template <class T, std::size_t N, class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, array<T, N> const& a)
{
    if (a.empty())
    {
        return os;
    }

    os << a[0];
    for (std::size_t i = 1; i < N; ++i)
    {
        os << ',' << a[i];
    }
    return os;
}

namespace detail
{
template <class T, std::size_t N, class Delimiter>
inline auto to_string(array<T, N> const& a, Delimiter const& delimiter)
    -> std::string
{
    if (a.empty())
        return {};

    std::ostringstream ss;

    ss << a[0];
    for (std::size_t i = 1; i < N; ++i)
    {
        ss << delimiter << a[i];
    }
    return ss.str();
};
} // namespace detail

template <class T, std::size_t N>
inline std::string to_string(array<T, N> const& a, char delimiter = ',')
{
    return detail::to_string(a, delimiter);
};

template <class T, std::size_t N>
inline std::string to_string(array<T, N> const& a, char const* delimiter = ",")
{
    return detail::to_string(a, delimiter);
};

template <class T, std::size_t N>
inline std::string to_string(array<T, N> const& a,
                             std::string const& delimiter = ",")
{
    return detail::to_string(a, delimiter);
};

} // namespace dabun
