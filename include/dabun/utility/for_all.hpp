// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/core.hpp"
#include "dabun/utility/vek.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <type_traits>

namespace dabun
{
namespace detail
{

template <std::size_t Dim, class Fn, class Int, std::size_t N>
DABUN_ALWAYS_INLINE void for_all_helper(vek<Int, N>&       var,
                                        vek<Int, N> const& begin,
                                        vek<Int, N> const& end, Fn const& fn)
{
    if constexpr (Dim + 1 == N)
    {
        for (var[Dim] = begin[Dim]; var[Dim] < end[Dim]; ++var[Dim])
        {
            vek<Int, N> const& cvar = var;
            fn(cvar);
        }
    }
    else
    {
        for (var[Dim] = begin[Dim]; var[Dim] < end[Dim]; ++var[Dim])
        {
            for_all_helper<Dim + 1>(var, begin, end, fn);
        }
    }
}

template <std::size_t Dim, class Fn, class Int, std::size_t N>
DABUN_ALWAYS_INLINE void for_all_helper(vek<Int, N>&       var,
                                        vek<Int, N> const& end, Fn const& fn)
{
    if constexpr (Dim + 1 == N)
    {
        for (var[Dim] = static_cast<Int>(0); var[Dim] < end[Dim]; ++var[Dim])
        {
            vek<Int, N> const& cvar = var;
            fn(cvar);
        }
    }
    else
    {
        for (var[Dim] = static_cast<Int>(0); var[Dim] < end[Dim]; ++var[Dim])
        {
            for_all_helper<Dim + 1>(var, end, fn);
        }
    }
}

} // namespace detail

template <class Fn, class Int, std::size_t N>
DABUN_ALWAYS_INLINE auto coord_for_loop(vek<Int, N> const& begin,
                                        vek<Int, N> const& end, Fn const& fn)
    -> std::enable_if_t<std::is_invocable_v<Fn const&, vek<Int, N> const&>>
{
    vek<Int, N> var;
    detail::for_all_helper<0>(var, begin, end, fn);
}

template <class Fn, class Int, std::size_t N>
DABUN_ALWAYS_INLINE auto coord_for_loop(vek<Int, N> const& end, Fn const& fn)
    -> std::enable_if_t<std::is_invocable_v<Fn const&, vek<Int, N> const&>>
{
    vek<Int, N> var;
    detail::for_all_helper<0>(var, end, fn);
}

} // namespace dabun
