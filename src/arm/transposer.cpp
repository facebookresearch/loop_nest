// Copyright 2004-present Facebook. All Rights Reserved.

#include "dabun/arm/transposer.hpp"

#ifdef DABUN_NOT_HEADER_ONLY

namespace dabun::arm
{

template class transposer_code_generator<aarch64, fp32>;
template class transposer_code_generator<aarch64, fp16>;

} // namespace dabun::arm

#endif
