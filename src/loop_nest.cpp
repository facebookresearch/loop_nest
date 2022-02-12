// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef DABUN_HEADER_ONLY

#include "dabun/loop_nest.hpp"

namespace dabun
{

#if defined(DABUN_ARCH_AARCH64)

namespace arm
{

template class loop_nest_code_generator<aarch64, true>;
template class loop_nest_code_generator<aarch64, false>;

} // namespace arm

#else

namespace x86
{

template class loop_nest_code_generator<avx2>;
template class loop_nest_code_generator<avx512>;
// template struct dabun::x86::loop_nest_code_generator<dabun::avx2_plus>;

} // namespace x86

#endif

} // namespace dabun

#endif
