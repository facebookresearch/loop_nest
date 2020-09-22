#pragma once

#include "dabun/aligned_vector.hpp"

#include <random>

namespace dabun
{

template <class Float>
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

} // namespace dabun
