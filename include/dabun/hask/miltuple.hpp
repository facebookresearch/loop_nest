// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#error "Deprecated"

#include <tuple>

namespace dabun::hask
{

namespace detail
{

template <std::size_t N, class T, class... Args>
struct miltuple_builder
{
    using type = typename miltuple_builder<N - 1, T, T, Args...>::type;
};

template <class T, class... Args>
struct miltuple_builder<0, T, Args...>
{
    using type = std::tuple<Args...>;
};

template <class T, std::size_t N>
struct miltuple_creator
{
    using type = typename miltuple_builder<N, T>::type;
};

} // namespace detail

template <class T, std::size_t N>
using miltuple = typename detail::miltuple_creator<T, N>::type;

} // namespace dabun::hask
