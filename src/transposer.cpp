// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef DABUN_HEADER_ONLY

#    include "dabun/transposer.hpp"

namespace dabun
{

#    if defined(DABUN_ARCH_AARCH64)

namespace arm
{

template class transposer_code_generator<aarch64, fp32_t>;
template class transposer_code_generator<aarch64, fp16_t>;

} // namespace arm

#    else

namespace x86
{

template class transposer_code_generator<avx2>;
template class transposer_code_generator<avx2_plus>;
template class transposer_code_generator<avx512>;

} // namespace x86

#    endif

} // namespace dabun

#endif
