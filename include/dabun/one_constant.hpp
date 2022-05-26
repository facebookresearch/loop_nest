// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace dabun
{

template <class Float>
Float const one_actual_constant = static_cast<Float>(1);

template <class Float>
Float const* const one_constant = &one_actual_constant<Float>;

} // namespace dabun
