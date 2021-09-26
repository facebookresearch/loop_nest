// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/third_party/biovault_bfloat16.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>

namespace dabun
{

using bfloat16_t = biovault::bfloat16_t;

class alignas(float) bf16x2_t
{
public:
    using value_type      = bfloat16_t;
    using underlying_type = std::uint32_t;
    using extended_type   = float;

private:
    union data_t
    {
        underlying_type           underlying_uint32_t;
        std::array<bfloat16_t, 2> values;

        constexpr data_t(underlying_type i)
            : underlying_uint32_t(i)
        {
        }
        data_t(float a, float b)
            : values({value_type(b), value_type(a)})
        {
        }
        constexpr data_t()
            : underlying_uint32_t(0)
        {
        }
    } data_;

public:
    bf16x2_t() = default;

    bf16x2_t(float a, float b)
        : data_(a, b)
    {
    }

    explicit bf16x2_t(underlying_type v) { data_.underlying_uint32_t = v; }

    constexpr bf16x2_t& operator=(bf16x2_t const&) = default;
    constexpr bf16x2_t& operator=(bf16x2_t&&) = default;

    constexpr bf16x2_t(bf16x2_t const&) = default;
    constexpr bf16x2_t(bf16x2_t&&)      = default;

    constexpr value_type& operator[](std::size_t i)
    {
        assert(i < 2);
        return data_.values[1 - i];
    }

    constexpr value_type const& operator[](std::size_t i) const
    {
        assert(i < 2);
        return data_.values[1 - i];
    }

    extended_type extended(std::size_t i) const
    {
        return static_cast<extended_type>(this->operator[](i));
    }

    constexpr underlying_type& underlying()
    {
        return data_.underlying_uint32_t;
    }

    constexpr underlying_type const& underlying() const
    {
        return data_.underlying_uint32_t;
    }
};

static_assert(sizeof(bf16x2_t) == 4);

inline constexpr float fmadd(bf16x2_t left, bf16x2_t right, float v)
{
    for (std::size_t i = 0; i < 2; ++i)
    {
        v += left.extended(i) * right.extended(i);
    }
    return v;
}

inline constexpr float dot(bf16x2_t left, bf16x2_t right)
{
    return fmadd(left, right, 0.f);
}

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, bf16x2_t const& v)
{
    os << "bf16x2_t<" << v.extended(0) << "," << v.extended(1) << ">";
    return os;
}

template <class T>
struct is_bf16x2_t : std::is_same<T, bf16x2_t>
{
};

template <class T>
inline constexpr bool is_bf16x2_v = is_bf16x2_t<T>::value;

} // namespace dabun
