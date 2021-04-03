#pragma once

#include "dabun/aligned_vector.hpp"
#include "dabun/qvec4.hpp"

#include <random>
#include <type_traits>

namespace dabun
{

template <class Float,
          class SFINAE = std::enable_if_t<std::is_floating_point_v<Float>>>
aligned_vector<Float> get_random_vector(unsigned size,
                                        unsigned extra_elements = 16)
{
    aligned_vector<Float> res(size + extra_elements);

    std::random_device rd;
    std::mt19937       gen(0); // rd());

    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (auto& f : res)
    {
        f = dis(gen);
    }

    return res;
}

template <class Float>
aligned_vector<Float> get_zero_vector(unsigned size,
                                      unsigned extra_elements = 16)
{
    aligned_vector<Float> res(size + extra_elements);
    return res;
}

template <class QVEC4, class SFINAE = std::enable_if_t<is_qvec4_v<QVEC4>>>
aligned_vector<QVEC4> get_random_qvector(unsigned size,
                                         unsigned extra_elements = 16)
{
    aligned_vector<QVEC4> res(size + extra_elements);

    std::random_device rd;
    std::mt19937       gen(0); // rd());

    std::uniform_int_distribution<std::uint32_t> dis(0, 0xffffffffu);

    for (auto& f : res)
    {
        f.underlying() = dis(gen);
    }

    return res;
}

template <class QVEC4, class SFINAE = std::enable_if_t<is_qvec4_v<QVEC4>>>
aligned_vector<QVEC4> get_zero_qvector(unsigned size,
                                       unsigned extra_elements = 16)
{
    aligned_vector<QVEC4> res(size + extra_elements);
    for (auto& v : res)
    {
        v.underlying() = 0u;
    }

    return res;
}

} // namespace dabun
