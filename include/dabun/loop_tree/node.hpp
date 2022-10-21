// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/code_generator/aot_fn.hpp"
#include "dabun/configuration.hpp"
#include "dabun/isa.hpp"
#include "dabun/loop_nest.hpp"
#include "dabun/loop_tree/report.hpp"
#include "dabun/loop_tree/types.hpp"
#include "dabun/loop_tree/utility.hpp"
#include "dabun/utility/log.hpp"

#include <map>
#include <string>
#include <vector>

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

inline std::map<node_kind, std::string> const node_kind_to_str_map = {
    {node_kind::for_loop, "for_loop_node"},
    {node_kind::compute, "compute_node"},
    {node_kind::transpose, "transpose_node"},
    {node_kind::compiled_loop_nest, "compiled_loop_nest_node"},
    {node_kind::compiled_transpose, "compiled_transpose_node"}};

inline std::string const& node_kind_to_str(node_kind kind)
{
    return node_kind_to_str_map.at(kind);
}

template <extension VEX, class Arithmetic>
class node
{

private:
    node_kind                              kind_;
    std::vector<node_ptr<VEX, Arithmetic>> children_;

public:
    virtual ~node(){};

    explicit node(node_kind kind)
        : kind_(kind)
    {
    }

    std::vector<node_ptr<VEX, Arithmetic>> const& get_children() const
    {
        return children_;
    }

    void set_children(std::vector<node_ptr<VEX, Arithmetic>> const& children)
    {
        children_ = children;
    }

    void set_children(std::vector<node_ptr<VEX, Arithmetic>>&& children)
    {
        children_ = std::move(children);
    }

    node_kind kind() const { return kind_; }

    // tensor positions, dimension sizes, and tensor formulas
    virtual std::pair<loop_tree_fn_type<Arithmetic>, report_vector>
    get_fn(std::map<std::string, int> const&, std::map<std::string, int> const&,
           std::map<std::string, int> const&, formulas_map_type const&,
           bool) const = 0;

    virtual std::set<std::string> get_tensors_used() const = 0;

    virtual std::set<std::string> get_output_tensors() const = 0;

    virtual strides_map_type const& get_tensor_strides() const = 0;

    virtual std::string dump(formulas_map_type const&          formulas,
                             std::map<std::string, int> const& sizes,
                             std::string const& indent) const = 0;
};

template <extension VEX, class Arithmetic>
class compute_node;

template <extension VEX, class Arithmetic>
class compiled_loop_nest_node;

template <extension VEX, class Arithmetic>
class transpose_node;

template <extension VEX, class Arithmetic>
class compiled_transpose_node;

template <extension VEX, class Arithmetic>
class for_loop_node;

template <extension VEX, class Arithmetic, std::size_t Cardinality>
class nested_for_loops_node;

} // namespace loop_tree
} // namespace dabun
