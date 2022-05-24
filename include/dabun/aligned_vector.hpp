// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sysml/memory.hpp>

#include <vector>

namespace dabun
{

template <typename T>
using aligned_vector = std::vector<T, sysml::aligned_allocator<T, 64>>;

} // namespace dabun
