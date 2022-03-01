// Copyright 2004-2021 Facebook Inc, 2021-present Meta Inc; All Rights Reserved.

#pragma once

#include "dabun/core.hpp"
#include "dabun/math.hpp"

#include <boost/compressed_pair.hpp>

#include <cstddef>
#include <type_traits>

namespace dabun
{

namespace detail
{

struct empty_type
{
};

template <std::size_t s>
using type_of_size = std::conditional_t<s == 0, empty_type, std::byte[s]>;

template <class T, std::size_t Alignment = std::alignment_of_v<T>,
          bool Padded = false>
class primitive_aligned_wrapper
{
private:
    using padding_type =
        std::conditional_t<Padded, type_of_size<round_up(sizeof(T), Alignment)>,
                           empty_type>;

private:
    alignas(Alignment) T t_;

    char pad[Alignment - sizeof(T) + 1];

    // int a, b, c;

public:
    constexpr primitive_aligned_wrapper() {}

    constexpr primitive_aligned_wrapper(T const& t)
        : t_(t)
    {
    }

    operator T&() { return t_; }

    constexpr operator T const&() const { return t_; }

    T const* operator&() const { return &t_; }
    T*       operator&() { return &t_; }
};
} // namespace detail

} // namespace dabun
