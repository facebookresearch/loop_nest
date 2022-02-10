// Copyright 2004-present Facebook. All Rights Reserved.

#include "dabun/arm/loop_nest.hpp"

#ifdef DABUN_NOT_HEADER_ONLY

namespace dabun::arm
{

template class loop_nest_code_generator<aarch64, true>;
template class loop_nest_code_generator<aarch64, false>;

} // namespace dabun::arm

#endif
