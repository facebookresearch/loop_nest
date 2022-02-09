// Copyright 2004-present Facebook. All Rights Reserved.

#include "dabun/x86/loop_nest.hpp"

#ifdef DABUN_NOT_HEADER_ONLY

template class dabun::x86::loop_nest_code_generator<dabun::avx2>;
template class dabun::x86::loop_nest_code_generator<dabun::avx512>;
// template struct dabun::x86::loop_nest_code_generator<dabun::avx2_plus>;

#endif
