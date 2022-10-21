// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "dabun/numeric.hpp"
#include "dabun/peak_gflops.hpp"

#include <iostream>

using namespace dabun;

#ifndef DABUN_ARITHMETIC
#    define DABUN_ARITHMETIC float
#endif

#ifndef DABUN_ISA
#    define DABUN_ISA avx2
#endif

int main()
{
    std::cout << measure_peak_gflops<DABUN_ISA, DABUN_ARITHMETIC>(1) << "\n";
}
