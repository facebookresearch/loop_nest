// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/numeric.hpp"
#include "sysml/math.hpp"

#include <cmath>
#include <iostream>

namespace dabun
{

template <class Float>
void apply_relu(Float* Begin, Float* End)
{
    for (; Begin != End; ++Begin)
    {
        if constexpr (std::is_same_v<Float, fp16_t>)
        {
            *Begin = static_cast<fp16_t>(
                std::max(static_cast<float>(0), static_cast<float>(*Begin)));
        }
        else
        {
            *Begin = std::max(static_cast<Float>(0), *Begin);
        }
    }
}

template <class Float>
auto max_abs_difference(Float const* LBegin, Float const* LEnd,
                        Float const* RBegin)
{
    decltype(sysml::absolute_difference(*LBegin, *RBegin)) res = 0;

    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        res = std::max(res, sysml::absolute_difference(*LBegin, *RBegin));
    }
    return res;
}

template <class Float>
Float max_abs_difference_verbose(Float const* LBegin, Float const* LEnd,
                                 Float const* RBegin)
{
    int   off = 0;
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        if constexpr (std::is_same_v<Float, fp16_t>)
        {
            std::cout << off++ << " : " << *LBegin << " "
                      << static_cast<float>(*RBegin) << " "
                      << std::abs(static_cast<float>(*LBegin - *RBegin))
                      << "\n";
            res = static_cast<fp16_t>(
                std::max(static_cast<float>(res),
                         std::abs(static_cast<float>(*LBegin) -
                                  static_cast<float>(*RBegin))));
        }
        else
        {
            std::cout << off++ << " : " << *LBegin << " " << *RBegin << " "
                      << std::abs(*LBegin - *RBegin) << "\n";
            res = std::max(res, std::abs(*LBegin - *RBegin));
        }
    }
    return res;
}

template <class Float>
Float max_abs_difference_verbose(Float const* LBegin, Float const* LEnd,
                                 Float const* RBegin, float delta)
{
    int   off = 0;
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        if (std::abs(*LBegin - *RBegin) > delta)
        {
            std::cout << off << " : " << *LBegin << " " << *RBegin << " "
                      << std::abs(*LBegin - *RBegin) << "\n";
        }
        res = std::max(res, std::abs(*LBegin - *RBegin));
        off++;
    }
    return res;
}

} // namespace dabun
