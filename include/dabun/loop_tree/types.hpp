// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/elementwise_operation.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dabun
{
namespace loop_tree
{

// forward declaration
template <class ISA>
class loop_tree_node;

// Type aliases for readability
// void (vector of tensors, vector of alpha offsets)
using loop_tree_fn_type =
    std::function<void(std::vector<float*> const&, std::vector<int> const&)>;

// map from name to map of strides
using strides_map_type = std::map<std::string, std::map<std::string, int>>;

// map from name to set of dimensions
using formulas_map_type = std::map<std::string, std::set<std::string>>;

template <class ISA>
using elementwise_op_ptr = std::shared_ptr<elementwise_operation<ISA>>;

template <class ISA>
using loop_tree_node_ptr = std::shared_ptr<loop_tree_node<ISA>>;

} // namespace loop_tree
} // namespace dabun
