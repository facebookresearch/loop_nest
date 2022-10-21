// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/aligned_vector.hpp"
#include "dabun/numeric.hpp"
#include "sysml/random.hpp"

#include <limits>
#include <random>
#include <type_traits>

namespace dabun
{

namespace detail
{

template <class T>
struct random_initalizer_helper
{

    template <class Float = T>
    static auto get_random_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<std::is_floating_point_v<Float> ||
                                std::is_same_v<Float, fp16_t>,
                            aligned_vector<Float>>
    {
        aligned_vector<Float> res(size + extra_elements);

        std::random_device rd;
        std::mt19937       gen(0); // rd());

        sysml::uniform_distribution<double> dis(-1.0, 1.0);

        for (auto& f : res)
        {
            f = dis(gen);
        }

        return res;
    }

    template <class Integer = T>
    static auto get_random_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<std::is_integral_v<Integer>,
                            aligned_vector<Integer>>
    {
        aligned_vector<Integer> res(size + extra_elements);

        std::random_device rd;
        std::mt19937       gen(0); // rd());

        sysml::uniform_distribution<Integer> dis(
            std::numeric_limits<Integer>::min(),
            std::numeric_limits<Integer>::max());

        for (auto& f : res)
        {
            f = dis(gen);
        }

        return res;
    }

    template <class Float = T>
    static auto get_zero_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<std::is_floating_point_v<Float> ||
                                std::is_same_v<Float, fp16_t>,
                            aligned_vector<Float>>
    {
        aligned_vector<Float> res(size + extra_elements);
        return res;
    }

    template <class Integer = T>
    static auto get_zero_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<std::is_integral_v<Integer>,
                            aligned_vector<Integer>>
    {
        aligned_vector<Integer> res(size + extra_elements);
        return res;
    }
};

} // namespace detail

template <class T>
decltype(auto) get_random_vector(unsigned size, unsigned extra_elements = 16)
{
    return detail::random_initalizer_helper<T>::get_random_vector(
        size, extra_elements);
}

template <class T>
decltype(auto) get_zero_vector(unsigned size, unsigned extra_elements = 16)
{
    return detail::random_initalizer_helper<T>::get_zero_vector(size,
                                                                extra_elements);
}

template <class To, class From>
auto aligned_vector_cast(aligned_vector<From> const& from)
    -> std::enable_if_t<std::is_convertible_v<From, To>, aligned_vector<To>>
{
    aligned_vector<To> ret(from.size());

    for (std::size_t i = 0; i < from.size(); ++i)
    {
        ret[i] = static_cast<To>(from[i]);
    }

    return ret;
}

} // namespace dabun
