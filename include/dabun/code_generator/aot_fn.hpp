// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sysml/code_generator/code_generated_fn.hpp>
#include <sysml/code_generator/protect.hpp>

namespace dabun
{

// using ::sysml::code_generator::code_generated_fn_ref;
using ::sysml::code_generator::observed_dynamic_fn;
using ::sysml::code_generator::shared_dynamic_fn;
using ::sysml::code_generator::unique_dynamic_fn;
using ::sysml::code_generator::weak_dynamic_fn;

using ::sysml::code_generator::dynamic_fn_cast;

} // namespace dabun
