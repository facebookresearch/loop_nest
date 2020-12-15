#pragma once

// From: https://github.com/herumi/xbyak/blob/master/xbyak/xbyak.h

#if defined(__APPLE__)

#include <sys/sysctl.h>

#ifndef MAP_JIT
#define MAP_JIT 0x800
#endif

namespace dabun
{
namespace detail
{

inline constexpr int mojave_version = 18;

inline int get_macOS_version_pure()
{
    char   buf[64];
    size_t size = sizeof(buf);
    int    err  = sysctlbyname("kern.osrelease", buf, &size, NULL, 0);
    if (err != 0)
    {
        return 0;
    }
    char* endp;
    int   major = strtol(buf, &endp, 10);
    if (*endp != '.')
    {
        return 0;
    }
    return major;
}

inline int get_macOS_version()
{
    static const int version = get_macOS_version_pure();
    return version;
}

} // namespace detail
} // namespace dabun

#endif
