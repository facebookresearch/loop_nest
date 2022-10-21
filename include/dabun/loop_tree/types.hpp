// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/elementwise_operation.hpp"
#include "dabun/isa.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace dabun
{
namespace loop_tree
{

// forward declaration
template <extension, class>
class node;

// Type aliases for readability
// void (vector of tensors, vector of alpha offsets)
template <class Arithmetic>
using loop_tree_fn_type =
    std::function<void(std::vector<Arithmetic*>&, std::vector<int>&)>;

// map from name to map of strides
using strides_map_type = std::map<std::string, std::map<std::string, int>>;

// map from name to set of dimensions
using formulas_map_type = std::map<std::string, std::set<std::string>>;

template <class ISA>
using elementwise_op_ptr = std::shared_ptr<elementwise_operation<ISA>>;

template <extension VEX, class Arithmetic>
using node_ptr = std::shared_ptr<node<VEX, Arithmetic>>;

// Note: add classes from dabun/arithmetic_operations.hpp
// as needed
enum class arithmetic_op_kind
{
    plus,
    multiplies,
    max,
    min
};

} // namespace loop_tree
} // namespace dabun
