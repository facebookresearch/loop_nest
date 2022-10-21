// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <atomic>
#include <sstream>
#include <string>
#include <thread>

namespace dabun::utility
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

} // namespace dabun::utility
