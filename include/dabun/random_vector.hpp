#pragma once

#include "sysml/random.hpp"
#include "dabun/aligned_vector.hpp"
#include "dabun/bf16x2.hpp"
#include "dabun/float.hpp"
#include "dabun/qvec4.hpp"

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
    template <class BFloatx2 = T>
    static auto get_random_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<is_bf16x2_v<BFloatx2>, aligned_vector<BFloatx2>>
    {
        aligned_vector<BFloatx2> res(size + extra_elements);

        std::random_device rd;
        std::mt19937       gen(0); // rd());

        sysml::uniform_distribution<float> dis(-1.0, 1.0);

        for (auto& f : res)
        {
            f[0] = dis(gen);
            f[1] = dis(gen);
        }

        return res;
    }

    template <class Float = T>
    static auto get_random_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<std::is_floating_point_v<Float> || is_fp16_v<Float>,
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

    template <class QVec4 = T>
    static auto get_random_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<is_qvec4_v<QVec4>, aligned_vector<QVec4>>
    {
        aligned_vector<QVec4> res(size + extra_elements);

        std::random_device rd;
        std::mt19937       gen(0); // rd());

        sysml::uniform_distribution<std::uint32_t> dis(0, 0xffffffffu);

        for (auto& f : res)
        {
            f.underlying() = dis(gen);
        }

        return res;
    }

    template <class Float = T>
    static auto get_zero_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<std::is_floating_point_v<Float> || is_fp16_v<Float>,
                            aligned_vector<Float>>
    {
        aligned_vector<Float> res(size + extra_elements);
        return res;
    }

    template <class BFloatx2 = T>
    static auto get_zero_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<is_bf16x2_v<BFloatx2>, aligned_vector<BFloatx2>>
    {
        aligned_vector<BFloatx2> res(size + extra_elements);
        for (auto& v : res)
        {
            v.underlying() = 0u;
        }
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

    template <class QVec4 = T>
    static auto get_zero_vector(unsigned size, unsigned extra_elements)
        -> std::enable_if_t<is_qvec4_v<QVec4>, aligned_vector<QVec4>>
    {
        aligned_vector<QVec4> res(size + extra_elements);
        for (auto& v : res)
        {
            v.underlying() = 0u;
        }

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
