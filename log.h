#pragma once

#include <iostream>

namespace facebook
{
namespace sysml
{
namespace aot
{

#ifndef NDEBUG
static constexpr bool DEBUG = true;
static constexpr bool INFO  = true;
#else
static constexpr bool DEBUG = false;
static constexpr bool INFO  = false;
#endif

class LN_LOG
{
private:
    bool print_ = false;

public:
    explicit LN_LOG(bool p)
        : print_(p)
    {
    }

    template <class T>
    LN_LOG const& operator<<(T&& v) const
    {
        if (print_)
        {
            std::cout << v;
        }
        return *this;
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
