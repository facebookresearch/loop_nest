// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/mpl/core.hpp"
#include <type_traits>

namespace dabun
{
namespace mpl
{

template <class...>
struct cond;

template <class T>
struct cond<else_condition, T>
{
    using type = T;
};

} // namespace mpl
} // namespace dabun
