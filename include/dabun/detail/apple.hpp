#pragma once

#if defined(__APPLE__)

#    include <cstddef>
#    include <cstdlib>
#    include <sys/sysctl.h>

#    ifndef MAP_JIT
#        define MAP_JIT 0x800
#    endif

namespace dabun
{
namespace detail
{

inline constexpr int mojave_version = 18;

inline int get_macOS_version()
{
    static const int version = []()
    {
        char        buffer[64];
        std::size_t size = sizeof(buffer);

        if (auto err =
                sysctlbyname("kern.osrelease", buffer, &size, nullptr, 0);
            err != 0)
        {
            return 0;
        }

        char* endp = nullptr;

        int ver_major = std::strtol(buffer, &endp, 10);

        if (*endp != '.')
        {
            return 0;
        }
        return ver_major;
    }();

    return version;
}

} // namespace detail
} // namespace dabun

#endif
