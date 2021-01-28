#pragma once

#include "dabun/float.hpp"

#include <cmath>
#include <iostream>

namespace dabun
{

template <class Float>
printable_fp_t<Float> max_abs_difference(Float const* LBegin, Float const* LEnd,
                                         Float const* RBegin)
{
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        res = std::max(res, std::abs(*LBegin - *RBegin));
    }
    return printable(res);
}

template <class Float>
printable_fp_t<Float> max_abs_difference_verbose(Float const* LBegin,
                                                 Float const* LEnd,
                                                 Float const* RBegin)
{
    int   off = 0;
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        std::cout << off++ << " : " << printable(*LBegin) << " "
                  << printable(*RBegin) << " "
                  << printable(std::abs(*LBegin - *RBegin)) << "\n";
        res = std::max(res, std::abs(*LBegin - *RBegin));
    }
    return printable(res);
}

template <class Float>
printable_fp_t<Float>
max_abs_difference_verbose(Float const* LBegin, Float const* LEnd,
                           Float const* RBegin, float delta)
{
    int   off = 0;
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        if (std::abs(*LBegin - *RBegin) > delta)
        {
            std::cout << off << " : " << printable(*LBegin) << " "
                      << printable(*RBegin) << " "
                      << printable(std::abs(*LBegin - *RBegin)) << "\n";
        }
        res = std::max(res, std::abs(*LBegin - *RBegin));
        off++;
    }
    return printable(res);
}

} // namespace dabun
