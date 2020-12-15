#pragma once

// From: https://github.com/herumi/xbyak/blob/master/xbyak/xbyak.h

#if defined(__APPLE__)
#define DABUN_USE_MAP_JIT
#include <sys/sysctl.h>
#ifndef MAP_JIT
#define MAP_JIT 0x800
#endif
#endif

namespace dabun
{
namespace detail
{

#if defined(DABUN_USE_MAP_JIT)

inline int get_macOS_version_pure()
{
    char   buf[64];
    size_t size = sizeof(buf);
    int    err  = sysctlbyname("kern.osrelease", buf, &size, NULL, 0);
    if (err != 0)
        return 0;
    char* endp;
    int   major = strtol(buf, &endp, 10);
    if (*endp != '.')
        return 0;
    return major;
}

inline int get_macOS_version()
{
    static const int version = get_macOS_version();
    return version;
}

#endif

} // namespace detail
} // namespace dabun
