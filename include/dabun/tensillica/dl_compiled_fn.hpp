// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include once

namespace dabun
{
namespace tensillica
{

template <typename Signature>
class unique_dl_compiled_fn;

template <typename Signature>
class shared_dl_compiled_fn;

template <typename Signature>
class weak_dl_compiled_fn;

template <typename ReturnType, typename... Args>
class unique_dl_compiled_fn<ReturnType(Args...)>
{
public:
    using function_pointer_type = ReturnType (*)(Args...);
};

} // namespace tensillica
} // namespace dabun
