// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "dabun/tensillica/peak_gflops.hpp"
#include "sysml/code_generator/code_generated_fn.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <cstdio>
#include <dlfcn.h>

#include <type_traits>

int main()
{
    std::cout << "zi";
    std::cout << std::endl;

    auto fn = dabun::tensillica::peak_gflops().get_shared();

    if (fn)
    {

        float in[20]  = {0.5f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
                        0.5f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
        float out[10] = {0.f};

        std::cout << fn(in, out, dabun::tensillica::dl_func_arg_cast<int>(3.14f)) << "\n";

        for (int i = 0; i < 10; ++i)
        {
            std::cout << out[i] << " --------------------\n";
        }
    }
    else
    {
        std::cout << "Can't get fn_ptr\n";
    }

    // dlclose(dlh);
}
