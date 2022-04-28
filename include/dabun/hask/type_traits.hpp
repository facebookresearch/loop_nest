// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <type_traits>

namespace dabun::hask
{

// Different than std::is_integral - mainly, it's an int type that can
// be randomly generated:
// https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
template <typename T>
struct is_int
    : std::integral_constant<
          bool, std::is_same_v<T, short> || std::is_same_v<T, int> ||
                    std::is_same_v<T, long> || std::is_same_v<T, long long> ||
                    std::is_same_v<T, unsigned short> ||
                    std::is_same_v<T, unsigned int> ||
                    std::is_same_v<T, unsigned long> ||
                    std::is_same_v<T, unsigned long long>>
{
};

template <typename T>
inline constexpr bool is_int_v = is_int<T>::value;

// Not really any different than std::is_floating_point, but here for
// completeness
template <typename T>
struct is_real : std::is_floating_point<T>
{
};

template <typename T>
inline constexpr bool is_real_v = is_real<T>::value;

template <typename T>
struct is_number : std::integral_constant<bool, is_int_v<T> || is_real_v<T>>
{
};

template <typename T>
inline constexpr bool is_number_v = is_number<T>::value;

} // namespace dabun::hask
