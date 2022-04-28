// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/hask/type_traits.hpp"

#include <limits>
#include <optional>
#include <random>

namespace dabun::hask
{

namespace detail
{

template <class T>
struct int_random_traits
{
    static_assert(is_int_v<T>);
    static constexpr T low  = std::numeric_limits<T>::min();
    static constexpr T high = std::numeric_limits<T>::max();

    using random_distribution = std::uniform_int_distribution<T>;
};

template <class T>
struct real_random_traits
{
    static_assert(is_real_v<T>);
    static constexpr T low  = static_cast<T>(-1);
    static constexpr T high = static_cast<T>(1);

    using random_distribution = std::uniform_real_distribution<T>;
};

template <class T>
struct random_traits
    : std::conditional_t<is_int_v<T>, int_random_traits<T>,
                         std::conditional_t<is_real_v<T>, real_random_traits<T>,
                                            std::void_t<T>>>
{
};

} // namespace detail

template <class T, bool RandomDeviceInit = true>
void random_fill(T* begin, T* end, std::optional<T> const& low = std::nullopt,
                 std::optional<T> const& high = std::nullopt) noexcept
{
    static_assert(is_number_v<T>);

    if constexpr (RandomDeviceInit)
    {
        std::random_device rd;
        std::mt19937       gen(rd());

        using traits = detail::random_traits<T>;

        typename traits::random_distribution dis(low ? *low : traits::low,
                                                 high ? *high : traits::high);

        for (; begin != end; ++begin)
        {
            *begin = dis(gen);
        }
    }
    else
    {
        std::mt19937 gen(0);

        using traits = detail::random_traits<T>;

        typename traits::random_distribution dis(low ? *low : traits::low,
                                                 high ? *high : traits::high);

        for (; begin != end; ++begin)
        {
            *begin = dis(gen);
        }
    }
}

} // namespace dabun::hask
