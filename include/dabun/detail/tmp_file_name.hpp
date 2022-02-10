#pragma once

#include <atomic>
#include <sstream>
#include <string>
#include <thread>

namespace dabun
{
namespace detail
{

inline std::string get_temporary_file_name(std::string const& suffix,
                                           std::string const& dir = "/tmp")
{
    static std::atomic<int> counter(0);

    std::ostringstream oss;
    oss << dir << "/" << std::this_thread::get_id() << "_" << (counter++)
        << suffix;
    return oss.str();
}

} // namespace detail
} // namespace dabun
