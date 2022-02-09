// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"
#ifdef DABUN_ARCH_X86_64

// Mosty from
// https://en.wikipedia.org/wiki/Denormal_number#Disabling_denormal_floats_at_the_code_level

#include <xmmintrin.h>

#define LN_MM_DENORMALS_ZERO_MASK 0x0040
#define LN_MM_DENORMALS_ZERO_ON 0x0040
#define LN_MM_DENORMALS_ZERO_OFF 0x0000

#define LN_MM_FLUSH_ZERO_MASK 0x8000
#define LN_MM_FLUSH_ZERO_ON 0x8000
#define LN_MM_FLUSH_ZERO_OFF 0x0000

#define LN_MM_SET_DENORMALS_ZERO_MODE(mode)                                    \
    _mm_setcsr((_mm_getcsr() & ~LN_MM_DENORMALS_ZERO_MASK) | (mode))

#define LN_MM_GET_DENORMALS_ZERO_MODE()                                        \
    (_mm_getcsr() & LN_MM_DENORMALS_ZERO_MASK)

#define LN_MM_SET_FLUSH_ZERO_MODE(mode)                                        \
    _mm_setcsr((_mm_getcsr() & ~LN_MM_FLUSH_ZERO_MASK) | (mode))

#define LN_MM_GET_FLUSH_ZERO_MODE() (_mm_getcsr() & LN_MM_FLUSH_ZERO_MASK)

namespace dabun::detail
{
class denormals_disabler
{
private:
    unsigned int previous_value;

public:
    denormals_disabler()
    {
        previous_value = _mm_getcsr();
        _mm_setcsr(previous_value | LN_MM_DENORMALS_ZERO_ON |
                   LN_MM_FLUSH_ZERO_ON);
    }

    ~denormals_disabler() { _mm_setcsr(previous_value); }

    denormals_disabler(denormals_disabler const&) = delete;
    denormals_disabler& operator=(denormals_disabler const&) = delete;
};

inline denormals_disabler denormals_disabler_instance;

} // namespace dabun::detail

#undef LN_MM_DENORMALS_ZERO_MASK
#undef LN_MM_DENORMALS_ZERO_ON
#undef LN_MM_DENORMALS_ZERO_OFF

#undef LN_MM_SET_DENORMALS_ZERO_MODE
#undef LN_MM_GET_DENORMALS_ZERO_MODE

#undef LN_MM_FLUSH_ZERO_MASK
#undef LN_MM_FLUSH_ZERO_ON
#undef LN_MM_FLUSH_ZERO_OFF

#undef LN_MM_SET_FLUSH_ZERO_MODE
#undef LN_MM_GET_FLUSH_ZERO_MODE

#endif
