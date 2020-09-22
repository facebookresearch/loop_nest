#pragma once

#include <stdexcept>

#define LOOP_NEST_STRINGIFY_0(s) #s
#define LOOP_NEST_STRINGIFY(s) LOOP_NEST_STRINGIFY_0(s)

#define strong_assert(condition)                                               \
    if (!(condition))                                                          \
    {                                                                          \
        throw std::runtime_error(LOOP_NEST_STRINGIFY(                          \
            condition) " failed file: " __FILE__                               \
                       " line: " LOOP_NEST_STRINGIFY((__LINE__)));             \
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
overloaded(Ts...) -> overloaded<Ts...>;

} // namespace dabun
