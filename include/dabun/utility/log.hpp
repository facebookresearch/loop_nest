// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fstream>
#include <iostream>

namespace dabun
{

#ifndef NDEBUG
static constexpr bool DEBUG = true;
static constexpr bool INFO  = true;
#else
static constexpr bool DEBUG = false;
static constexpr bool INFO  = false;
#endif

#if defined(DABUN_LOG_TO_FILE)

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
        static std::ofstream fout("dabun_loop_nest.log");
        if (print_)
        {
            fout << v;
        }
        return *this;
    }
};

#else

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

#endif

} // namespace dabun
