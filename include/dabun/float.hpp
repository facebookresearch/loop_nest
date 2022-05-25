// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <sysml/numeric.hpp>
#include <type_traits>

namespace dabun
{

using fp32 = float;

template <class>
struct is_fp16_t : std::false_type
{
};

using fp16 = sysml::fp16;

template <>
struct is_fp16_t<fp16> : std::true_type
{
};

template <class T>
inline constexpr bool is_fp16_v = is_fp16_t<T>::value;

} // namespace dabun
