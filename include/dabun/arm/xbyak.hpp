// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"
#ifdef DABUN_ARCH_AARCH64

#    include "xbyak_aarch64/xbyak_aarch64.h"

#    include "dabun/core.hpp"

using xbyak_buffer_type = std::uint32_t;

namespace Xbyak
{
using namespace Xbyak_aarch64;
using CodeArray     = CodeArrayAArch64;
using Allocator     = AllocatorAArch64;
using CodeGenerator = CodeGeneratorAArch64;
using Reg64         = XReg;
using Label         = LabelAArch64;
} // namespace Xbyak

namespace dabun
{

template <unsigned ElementSize, unsigned NumElements = 16 / ElementSize>
struct vreg_view
{
private:
    static_assert(ElementSize == 1 || ElementSize == 2 || ElementSize == 4 ||
                  ElementSize == 8);

public:
    decltype(auto) operator()(Xbyak::VReg const& r)
    {
        if constexpr (ElementSize == 1)
        {
            if constexpr (NumElements == 4)
            {
                return r.b4;
            }
            else if constexpr (NumElements == 8)
            {
                return r.b8;
            }
            else if constexpr (NumElements == 16)
            {
                return r.b16;
            }
            else
            {
                strong_assert(false);
                return nullptr;
            }
        }
        else if constexpr (ElementSize == 2)
        {
            if constexpr (NumElements == 2)
            {
                return r.h2;
            }
            else if constexpr (NumElements == 4)
            {
                return r.h4;
            }
            else if constexpr (NumElements == 8)
            {
                return r.h8;
            }
            else
            {
                strong_assert(false);
                return nullptr;
            }
        }
        else if constexpr (ElementSize == 4)
        {
            // if constexpr (NumElements == 1)
            // {
            //     return r.s1;
            // }
            //else
            if constexpr (NumElements == 2)
            {
                return r.s2;
            }
            else if constexpr (NumElements == 4)
            {
                return r.s4;
            }
            else
            {
                strong_assert(false);
                return nullptr;
            }
        }
        else if constexpr (ElementSize == 8)
        {
            if constexpr (NumElements == 1)
            {
                return r.d1;
            }
            else if constexpr (NumElements == 2)
            {
                return r.d2;
            }
            else
            {
                strong_assert(false);
                return nullptr;
            }
        }
        else
        {
            strong_assert(false);
            return nullptr;
        }
    }
};

} // namespace dabun

#endif
