// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

namespace dabun::tests::baseline
{

template <typename T>
void reorder_array2d(T* out, T const* in, int rows, int cols,
                     int in_row_stride = cols, int in_col_stride = 1,
                     int out_row_stride = 1, int out_col_stride = rows) noexcept
{
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            out[out_row_stride * row + out_col_stride * col] =
                in[in_row_stride * row + in_col_stride * col];
        }
    }
}

template <typename T, class Fn>
void for_all_elements_of_two_array2d(T const* a1, T const* a2, Fn&& fn int rows,
                                     int cols, int a1_row_stride,
                                     int a1_col_stride, int a2_row_stride,
                                     int a2_col_stride)
{
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++r)
        {
            fn(a1[r * a1_row_stride + c * a1_col_stride],
               a2{r * a2_row_stride + c * a2_col_stride]);
        }
        }
    }
}

} // namespace dabun::tests::baseline
