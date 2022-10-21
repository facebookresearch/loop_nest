// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <iostream>

#include "dabun/tensillica/cpp_intrinsics_code_generator.hpp"

// #include "dabun/isa.hpp"
// #ifdef DABUN_ARCH_AARCH64

// #    include "dabun/code_generator/code_generator.hpp"
// #    include "dabun/isa.hpp"
// #    include "dabun/math.hpp"
// #    include "dabun/numeric.hpp"

// #    include <sysml/measure.hpp>

#include <utility>

namespace dabun
{
namespace tensillica
{

struct peak_gflops : cpp_intrinsics_code_generator<std::uint64_t(float*, float*, std::uint64_t)>
{
    peak_gflops()
    {
        multi_vmm<Vmm, SReg, Reg32> mvmm(8, 0);
        mvmm.full_reduce(*this, 4, 0);

        ldp(vmm0.s4, vmm1.s4, pre_ptr(x0, 4));

        mov(vmm1.s4, vmm0.s4);
        fmla(vmm1.s4, vmm1.s4, vmm0.s[1]);


        // ins(vmm0.s[1], w2);
        stp(vmm0.s4, vmm1.s4, ptr(x1));

        custom_string("return x0 + x1 + x2;");
    }
};

struct peak_gflopsw : cpp_intrinsics_code_generator<std::uint64_t(float*, float*, std::uint64_t)>
{
    peak_gflopsw()
    {
        multi_vmm<Vmm, SReg, Reg32> mvmm(8, 0);
        mvmm.full_reduce(*this, 4, 0);
        custom_string("return x0 * x1 * x2;");
    }
};



} // namespace tensillica
} // namespace dabun
