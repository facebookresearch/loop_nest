// Copyright 2004-present Facebook. All Rights Reserved.

#include "dabun/x86/transposer.hpp"

#ifdef DABUN_NOT_HEADER_ONLY

template class dabun::x86::transposer_code_generator<dabun::avx2>;
template class dabun::x86::transposer_code_generator<dabun::avx2_plus>;
template class dabun::x86::transposer_code_generator<dabun::avx512>;

#endif
