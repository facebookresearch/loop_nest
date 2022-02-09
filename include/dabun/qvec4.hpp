// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <type_traits>

namespace dabun
{

template <bool IsSigned>
class alignas(std::int32_t) qvec4
{
public:
    using value_type = std::conditional_t<IsSigned, std::int8_t, std::uint8_t>;
    using underlying_type = std::uint32_t;
    using extended_type   = std::int32_t;

private:
    union
    {
        underlying_type underlying_uint32_t;
        value_type      values[4];
    } data_;

public:
    constexpr qvec4() = default;

    constexpr qvec4(value_type v0, value_type v1, value_type v2, value_type v3)
    {
        data_.values[3] = v0;
        data_.values[2] = v1;
        data_.values[1] = v2;
        data_.values[0] = v3;
    }

    explicit qvec4(underlying_type v) { data_.underlying_uint32_t = v; }

    constexpr qvec4& operator=(qvec4 const&) = default;
    constexpr qvec4& operator=(qvec4&&) = default;

    constexpr qvec4(qvec4 const&) = default;
    constexpr qvec4(qvec4&&)      = default;

    constexpr value_type& operator[](std::size_t i)
    {
        assert(i < 4);
        return data_.values[3 - i];
    }

    constexpr value_type const& operator[](std::size_t i) const
    {
        assert(i < 4);
        return data_.values[3 - i];
    }

    constexpr extended_type extended(std::size_t i) const
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

template <bool LeftSigned, bool RightSigned>
constexpr std::int32_t vnni_fma(qvec4<LeftSigned>  left,
                                qvec4<RightSigned> right, std::int32_t v)
{
    for (std::size_t i = 0; i < 4; ++i)
    {
        v += left.extended(i) * right.extended(i);
    }
    return v;
}

template <bool LeftSigned, bool RightSigned>
constexpr std::int32_t dot(qvec4<LeftSigned> left, qvec4<RightSigned> right)
{
    return vnni_fma(left, right, 0);
}

template <bool IsSigned, class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, qvec4<IsSigned> const& v)
{
    os << (IsSigned ? "int8x4_t<" : "uint8x4_t<") << v.extended(0);
    for (std::size_t i = 1; i < 4; ++i)
    {
        os << "," << v.extended(i);
    }
    os << ">";
    return os;
}

using int8x4_t  = qvec4<true>;
using uint8x4_t = qvec4<false>;

static_assert(sizeof(int8x4_t) == 4);
static_assert(sizeof(uint8x4_t) == 4);

template <class>
struct is_qvec4 : std::false_type
{
};

template <bool IsSigned>
struct is_qvec4<qvec4<IsSigned>> : std::true_type
{
};

template <class T>
inline constexpr bool is_qvec4_v = is_qvec4<T>::value;

} // namespace dabun
