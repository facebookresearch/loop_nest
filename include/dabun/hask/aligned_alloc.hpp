// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

namespace dabun::hask
{

namespace detail
{

inline void* aligned_alloc_impl(std::size_t alignment, std::size_t size)
{
    // Workaround c++17 alligned_alloc (will get it from C11's stdlib.h header
    // otherwise);
    using namespace std;

    return aligned_alloc(alignment, size);
}

} // namespace detail

inline void* aligned_alloc(std::size_t alignment, std::size_t size)
{
    return detail::aligned_alloc_impl(alignment, size);
}

inline void* checked_aligned_alloc(std::size_t alignment, std::size_t size)
{
    auto ret = aligned_alloc(alignment, size);

    if (!ret)
    {
        throw std::bad_alloc();
    }

    return ret;
}


inline void aligned_free(void* ptr) noexcept
{
    return std::free(ptr);
}


} // namespace dabun::hask
