// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <map>
#include <string>

namespace dabun
{
namespace loop_tree
{

enum class node_kind
{
    for_loop,
    compute,
    transpose,
    compiled_loop_nest,
    compiled_transpose
};

inline std::map<node_kind, std::string> node_kind_to_str = {
    {node_kind::for_loop, "for_loop_node"},
    {node_kind::compute, "compute_node"},
    {node_kind::transpose, "transpose_node"},
    {node_kind::jitted_loop_nest, "jitted_loop_nest_node"},
    {node_kind::jitted_transpose, "jitted_transpose_node"}};

inline std::string const& node_kind_to_string(node_kind kind)
{
    return node_kind_to_str.at(kind);
}

// Note: add classes from arithmetic_operations.h
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
