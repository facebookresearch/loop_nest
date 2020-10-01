#pragma once

#include <stdexcept>

#define DABUN_STRINGIFY_0(s) #s
#define DABUN_STRINGIFY(s) DABUN_STRINGIFY_0(s)

#define strong_assert(condition)                                               \
    if (!(condition))                                                          \
    {                                                                          \
        throw std::runtime_error(                                              \
            DABUN_STRINGIFY(condition) " failed file: " __FILE__               \
                                       " line: " DABUN_STRINGIFY((__LINE__))); \
    }                                                                          \
    static_cast<void>(0)

namespace dabun
{

// FROM: https://en.cppreference.com/w/cpp/utility/variant/visit

template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};

// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...)->overloaded<Ts...>;

} // namespace dabun
