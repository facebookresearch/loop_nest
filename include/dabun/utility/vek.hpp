// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/utility/array.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace dabun
{

template <class T, std::size_t N>
class vek : public array<T, N>
{
private:
    using value_type = std::decay_t<T>;

    static_assert(std::is_same_v<T, value_type>);
    static_assert(std::is_arithmetic_v<T>);
    static_assert(N <= 64);

private:
    using super_type = std::array<T, N>;

public:
#define DABUN_UTILITY_VEK_COMPOUND_OPERATOR_DEFINITION(OP)                     \
    template <class U>                                                         \
    inline constexpr auto operator OP(vek<U, N> const& rhs) noexcept           \
        ->std::enable_if_t<std::is_scalar_v<U> && std::is_convertible_v<U, T>, \
                           vek<T, N>&>                                         \
    {                                                                          \
        for (std::size_t i = 0; i < N; ++i)                                    \
        {                                                                      \
            (*this)[i] OP rhs[i];                                              \
        }                                                                      \
        return *this;                                                          \
    }                                                                          \
                                                                               \
    template <class U>                                                         \
    inline constexpr auto operator OP(U rhs) noexcept->std::enable_if_t<       \
        std::is_scalar_v<U> && std::is_convertible_v<U, T>, vek<T, N>&>        \
    {                                                                          \
        for (std::size_t i = 0; i < N; ++i)                                    \
        {                                                                      \
            (*this)[i] OP rhs;                                                 \
        }                                                                      \
        return *this;                                                          \
    }

    DABUN_UTILITY_VEK_COMPOUND_OPERATOR_DEFINITION(+=)
    DABUN_UTILITY_VEK_COMPOUND_OPERATOR_DEFINITION(-=)
    DABUN_UTILITY_VEK_COMPOUND_OPERATOR_DEFINITION(*=)
    DABUN_UTILITY_VEK_COMPOUND_OPERATOR_DEFINITION(/=)
    DABUN_UTILITY_VEK_COMPOUND_OPERATOR_DEFINITION(%=)

#undef DABUN_UTILITY_VEK_COMPOUND_OPERATOR_DEFINITION

    template <class U>
    inline constexpr friend auto operator==(vek<T, N> const& lhs,
                                            vek<U, N> const& rhs) noexcept
        -> bool
    {
        return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs));
    }

    template <class U>
    inline constexpr friend auto operator!=(vek<T, N> const& lhs,
                                            vek<U, N> const& rhs) noexcept
        -> bool
    {
        return !(lhs == rhs);
    }
};

template <class T, std::size_t N_Lhs, std::size_t N_Rhs>
inline constexpr auto concat(vek<T, N_Lhs> const& lhs,
                             vek<T, N_Rhs> const& rhs) noexcept
    -> vek<T, N_Lhs + N_Rhs>
{
    vek<T, N_Lhs + N_Rhs> ret;

    std::copy_n(std::begin(lhs), N_Lhs, std::begin(ret));
    std::copy_n(std::begin(rhs), N_Rhs, std::begin(ret) + N_Lhs);

    return ret;
}

template <class T, std::size_t N_Lhs, std::size_t N_Rhs>
inline constexpr auto cat(vek<T, N_Lhs> const& lhs,
                          vek<T, N_Rhs> const& rhs) noexcept
    -> vek<T, N_Lhs + N_Rhs>
{
    return concat(lhs, rhs);
}

#define DABUN_UTILITY_VEK_BINARY_OPERATOR_DEFINITION(OP)                       \
                                                                               \
    template <class TyLhs, class TyRhs, std::size_t N>                         \
    inline constexpr auto operator OP(vek<TyLhs, N> const& lhs,                \
                                      vek<TyRhs, N> const& rhs) noexcept       \
        ->vek<DABUN_OP_RESULT_TYPE(OP, TyLhs, TyRhs), N>                       \
    {                                                                          \
        vek<DABUN_OP_RESULT_TYPE(OP, TyLhs, TyRhs), N> ret;                    \
        for (std::size_t i = 0; i < N; ++i)                                    \
        {                                                                      \
            ret[i] = lhs[i] OP rhs[i];                                         \
        }                                                                      \
        return ret;                                                            \
    }                                                                          \
                                                                               \
    template <class TyLhs, std::size_t N, class TyRhs>                         \
    inline constexpr auto operator OP(vek<TyLhs, N> const& lhs,                \
                                      TyRhs                rhs) noexcept                      \
        ->std::enable_if_t<std::is_scalar_v<TyRhs> &&                          \
                               std::is_convertible_v<TyRhs, TyLhs>,            \
                           vek<DABUN_OP_RESULT_TYPE(OP, TyLhs, TyRhs), N>>     \
    {                                                                          \
        vek<DABUN_OP_RESULT_TYPE(OP, TyLhs, TyRhs), N> ret;                    \
        for (std::size_t i = 0; i < N; ++i)                                    \
        {                                                                      \
            ret[i] = lhs[i] OP rhs;                                            \
        }                                                                      \
        return ret;                                                            \
    }                                                                          \
                                                                               \
    template <class TyLhs, class TyRhs, std::size_t N>                         \
    inline constexpr auto operator OP(TyLhs                lhs,                \
                                      vek<TyRhs, N> const& rhs) noexcept       \
        ->std::enable_if_t<std::is_scalar_v<TyLhs> &&                          \
                               std::is_convertible_v<TyLhs, TyRhs>,            \
                           vek<DABUN_OP_RESULT_TYPE(OP, TyLhs, TyRhs), N>>     \
    {                                                                          \
        vek<DABUN_OP_RESULT_TYPE(OP, TyLhs, TyRhs), N> ret;                    \
        for (std::size_t i = 0; i < N; ++i)                                    \
        {                                                                      \
            ret[i] = lhs OP rhs[i];                                            \
        }                                                                      \
        return ret;                                                            \
    }

DABUN_UTILITY_VEK_BINARY_OPERATOR_DEFINITION(+)
DABUN_UTILITY_VEK_BINARY_OPERATOR_DEFINITION(-)
DABUN_UTILITY_VEK_BINARY_OPERATOR_DEFINITION(/)
DABUN_UTILITY_VEK_BINARY_OPERATOR_DEFINITION(*)
DABUN_UTILITY_VEK_BINARY_OPERATOR_DEFINITION(%)

#undef DABUN_UTILITY_VEK_BINARY_OPERATOR_DEFINITION

template <class T, std::size_t N>
inline constexpr auto operator-(vek<T, N> const& rhs) noexcept
    -> std::enable_if_t<std::is_signed_v<T>, vek<T, N>>
{
    vek<T, N> ret;
    for (std::size_t i = 0; i < N; ++i)
    {
        ret[i] = -rhs[i];
    }
    return ret;
}

template <class T, std::size_t N>
inline constexpr vek<T, N> const& operator+(vek<T, N> const& rhs) noexcept
{
    return rhs;
}

// template <class T, std::size_t N, class CharT, class Traits>
// std::basic_ostream<CharT, Traits>&
// operator<<(std::basic_ostream<CharT, Traits>& os, vek<T, N> const& v)
// {
//     os << v[0];
//     for (std::size_t i = 1; i < N; ++i)
//     {
//         os << ',' << v[i];
//     }
//     return os;
// }

template <class T, class U, std::size_t N>
constexpr std::uint64_t
elementwise_compare_to_bitmask(vek<T, N> const& lhs,
                               vek<U, N> const& rhs) noexcept
{
    std::uint64_t ret = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        ret <<= 1;
        ret |= (lhs[i] == rhs[i] ? 1 : 0);
    }
    return ret;
}

template <std::size_t Len, class T, std::size_t N>
constexpr auto head(vek<T, N> const& v) noexcept
    -> std::enable_if_t<Len <= N, vek<T, Len>>
{
    vek<T, Len> ret;

    std::copy_n(std::begin(v), Len, std::begin(ret));

    return ret;
}

} // namespace dabun
