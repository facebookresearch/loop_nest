// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "dabun/arithmetic_operation.hpp"
#include "dabun/configuration.hpp"
#include "dabun/elementwise_operation.hpp"
#include "dabun/log.hpp"
#include "dabun/loop_nest.hpp"
#include "dabun/serialization.hpp"
#include "dabun/transposer.hpp"

#include "dabun/loop_tree/node.hpp"
#include "dabun/loop_tree/types.hpp"
#include "dabun/loop_tree/utility.hpp"

namespace dabun
{
namespace loop_tree
{

template <class Arithmetic>
class program
{
private:
    std::shared_ptr<node_report>                                   report;
    std::function<void(std::map<std::string, Arithmetic*> const&)> fn;

public:
    program(){};
    program(program const&) = default;
    program(program&&)      = default;
    program& operator=(program const&) = default;
    program& operator=(program&&) = default;

    template <class FN>
    program(std::shared_ptr<node_report> const& r, FN&& f)
        : report(r)
        , fn(f)
    {
    }

    void operator()(std::map<std::string, Arithmetic*> const& m) const
    {
        fn(m);
    }

    std::shared_ptr<node_report> const& get_report() const { return report; }
};

template <extension VEX, class Arithmetic>
node_ptr<VEX, Arithmetic> merge_loop_into_compiler(
    std::shared_ptr<for_loop_node<VEX, Arithmetic>> const&           node,
    std::shared_ptr<compiled_loop_nest_node<VEX, Arithmetic>> const& child)
{
    return node_ptr<VEX, Arithmetic>(
        new compiled_loop_nest_node<VEX, Arithmetic>(node, child));
}

template <extension VEX, class Arithmetic>
node_ptr<VEX, Arithmetic> merge_loop_into_compiler(
    std::shared_ptr<for_loop_node<VEX, Arithmetic>> const& node,
    std::shared_ptr<compute_node<VEX, Arithmetic>> const&  child)
{
    return node_ptr<VEX, Arithmetic>(
        new compiled_loop_nest_node<VEX, Arithmetic>(node, child));
}

template <extension VEX, class Arithmetic>
node_ptr<VEX, Arithmetic> merge_loop_into_compiler(
    std::shared_ptr<for_loop_node<VEX, Arithmetic>> const&  node,
    std::shared_ptr<transpose_node<VEX, Arithmetic>> const& child)
{
    return node_ptr<VEX, Arithmetic>(
        new compiled_transpose_node<VEX, Arithmetic>(node, child));
}

template <extension VEX, class Arithmetic>
node_ptr<VEX, Arithmetic> merge_loop_into_compiler(
    std::shared_ptr<for_loop_node<VEX, Arithmetic>> const&           node,
    std::shared_ptr<compiled_transpose_node<VEX, Arithmetic>> const& child)
{
    return node_ptr<VEX, Arithmetic>(
        new compiled_transpose_node<VEX, Arithmetic>(node, child));
}

template <extension VEX, class Arithmetic>
node_ptr<VEX, Arithmetic>
simplify_loop_nests(node_ptr<VEX, Arithmetic> const& node,
                    int current_depth = 0, int max_interpreted_depth = 0)
{
    if (node->kind() != node_kind::for_loop)
    {
        return node;
    }

    std::vector<node_ptr<VEX, Arithmetic>> new_children;
    for (auto c : node->get_children())
    {
        new_children.push_back(
            simplify_loop_nests(c, current_depth + 1, max_interpreted_depth));
    }
    node->set_children(new_children);

    // can't merge into loop nest compute or loop nest transpose
    // since has "split"
    if (new_children.size() > 1)
    {
        return node;
    }

    if (current_depth < max_interpreted_depth)
    {
        // part of the prefix we want to have interpreted
        return node;
    }

    auto for_node =
        std::dynamic_pointer_cast<for_loop_node<VEX, Arithmetic>>(node);
    node_ptr<VEX, Arithmetic> single_child = new_children.at(0);

    switch (single_child->kind())
    {
    case node_kind::for_loop:
        // child is not compiled, so can't add on this node
        return node;
        break;

    case node_kind::compute:
        return merge_loop_into_compiler(
            for_node, std::dynamic_pointer_cast<compute_node<VEX, Arithmetic>>(
                          single_child));
        break;

    case node_kind::compiled_loop_nest:
        return merge_loop_into_compiler(
            for_node,
            std::dynamic_pointer_cast<compiled_loop_nest_node<VEX, Arithmetic>>(
                single_child));
        break;

    case node_kind::transpose:
        return merge_loop_into_compiler(
            for_node,
            std::dynamic_pointer_cast<transpose_node<VEX, Arithmetic>>(
                single_child));
        break;

    case node_kind::compiled_transpose:
        return merge_loop_into_compiler(
            for_node,
            std::dynamic_pointer_cast<compiled_transpose_node<VEX, Arithmetic>>(
                single_child));
        break;

    default:
        throw std::runtime_error("Unhandled node kind");
    }
}

template <extension VEX, class Arithmetic>
inline std::string dump_recursively(node_ptr<VEX, Arithmetic> const&  node,
                                    formulas_map_type const&          formulas,
                                    std::map<std::string, int> const& sizes,
                                    std::string&                      indent)
{
    std::ostringstream ss;
    ss << node->dump(formulas, sizes, indent);
    ss << std::endl;
    if (node->kind() == node_kind::for_loop)
    {
        std::string next_indent = indent;
        next_indent += "  ";
        for (auto const& c : node->get_children())
        {
            ss << dump_recursively(c, formulas, sizes, next_indent);
        }
    }
    return ss.str();
}

inline std::int64_t get_tensor_num_elements(
    std::string const& name, strides_map_type const& strides,
    std::map<std::string, int> const& sizes, formulas_map_type const& formulas)
{
    std::int64_t size = 1;
    for (auto const& s : sizes)
    {
        if (formulas.at(name).count(s.first))
            size += (s.second - 1) * strides.at(name).at(s.first);
    }
    // size *= 4;
    return size;
}

template <extension VEX, class Arithmetic>
std::int64_t get_largest_intermediate_output_size(
    node_ptr<VEX, Arithmetic> const&  node,
    std::vector<std::string> const&   provided_tensors,
    std::map<std::string, int> const& sizes, formulas_map_type const& formulas)
{
    std::int64_t max_size = 0;
    switch (node->kind())
    {
    case node_kind::for_loop:
        for (auto const& child : node->get_children())
        {
            max_size =
                std::max(max_size, get_largest_intermediate_output_size(child));
        }
        return max_size;
        break;
    case node_kind::compute:
        // fall through
    case node_kind::transpose:
        // fall through
    case node_kind::compiled_loop_nest:
        // fall through
    case node_kind::compiled_transpose:
        std::set<std::string> possible_intermediates =
            node->get_output_tensors();

        for (auto const& name : provided_tensors)
        {
            possible_intermediates.erase(name);
        }

        for (auto const& name : possible_intermediates)
        {
            max_size = std::max(max_size, get_tensor_num_elements(
                                              name, node->get_tensor_strides(),
                                              sizes, formulas) *
                                              sizeof(Arithmetic));
        }

        return max_size;
        break;
        // default:
        // throw std::runtime_error("Unhandled node kind");
    }
}

template <extension VEX, class Arithmetic>
class loop_tree_program
{
private:
    std::vector<node_ptr<VEX, Arithmetic>> nodes;
    std::map<std::string, int>             sizes;
    formulas_map_type                      formulas;
    // for forcing partially interpreted trees (mainly for testing)
    int max_interpreted_depth;

    // map tensor names to indices
    std::map<std::string, int> tensors_idx;

public:
    loop_tree_program(std::vector<node_ptr<VEX, Arithmetic>> const& nodes,
                      std::map<std::string, int> const&             sizes,
                      formulas_map_type const&                      formulas,
                      std::optional<int> max_interpreted_depth = std::nullopt)
        : nodes(nodes)
        , sizes(sizes)
        , formulas(formulas)
        , max_interpreted_depth(max_interpreted_depth ? *max_interpreted_depth
                                                      : 0)
    {

        LN_LOG(DEBUG) << "Tree dump:\n";
        LN_LOG(DEBUG) << dump();

        LN_LOG(DEBUG) << "Pass: Simplifying loop nests\n";
        std::vector<node_ptr<VEX, Arithmetic>> new_nodes;
        for (auto c : nodes)
        {
            new_nodes.push_back(
                simplify_loop_nests(c, 0, this->max_interpreted_depth));
        }
        this->nodes = new_nodes;

        LN_LOG(DEBUG) << "Tree dump:\n";
        LN_LOG(DEBUG) << dump();

        LN_LOG(DEBUG) << "Pass: Map tensor names to indices in vector\n";
        int idx = 0;
        for (auto const& c : this->nodes)
        {
            // all (used) tensors need to provide strides
            // so this covers everything we need to map
            for (auto const& t : c->get_tensor_strides())
            {
                if (tensors_idx.count(t.first) == 0)
                {
                    tensors_idx[t.first] = idx;
                    idx += 1;
                }
            }
        };
    }

    std::vector<node_ptr<VEX, Arithmetic>> const& get_children()
    {
        return nodes;
    }

    std::string dump() const
    {
        std::ostringstream ss;
        std::string        indent = "";
        for (auto const& c : nodes)
        {
            ss << dump_recursively(c, formulas, sizes, indent);
        }
        return ss.str();
    }

    std::int64_t
    get_scratch_size(std::set<std::string> const& provided_tensors) const
    {
        std::int64_t max_size = 0;
        for (auto const& child : nodes)
        {
            max_size = std::max(max_size,
                                get_largest_intermediate_output_size(
                                    child, provided_tensors, sizes, formulas));
        }
        return max_size;
    }

    program<Arithmetic> get_fn(bool spit_asm = true) const
    {
        std::vector<loop_tree_fn_type> sub_functions;
        // added to alpha at runtime to handle tensor initialization
        int alpha_offsets_size = static_cast<int>(tensors_idx.size());

        std::map<std::string, int> iteration_depths;

        report_vector report;

        for (auto const& c : this->nodes)
        {
            auto sub = c->get_fn(tensors_idx, sizes, iteration_depths, formulas,
                                 spit_asm);
            sub_functions.push_back(sub.first);
            report.insert(report.end(), sub.second.begin(), sub.second.end());
        }

        return program<Arithmetic>(
            std::make_shared<node_report>(program_node_info{0, 0},
                                          std::move(report)),
            [sub_functions, alpha_offsets_size,
             tensors_idx = this->tensors_idx](
                std::map<std::string, Arithmetic*> const& tensors) {
                std::vector<int>         alpha_offs(alpha_offsets_size);
                std::vector<Arithmetic*> tensors_vec(tensors_idx.size());
                for (auto const& e : tensors)
                {
                    int idx          = tensors_idx.at(e.first);
                    tensors_vec[idx] = e.second;
                }

                for (auto const& f : sub_functions)
                {
                    f(tensors_vec, alpha_offs);
                }
            });
    }
};

template <extension VEX, class Arithmetic>
std::shared_ptr<loop_tree_program<VEX, Arithmetic>>
make_loop_tree_program(std::vector<node_ptr<VEX, Arithmetic>> const& nodes,
                       std::map<std::string, int> const&             sizes,
                       formulas_map_type const&                      formulas,
                       std::optional<int> max_interpreted_depth = std::nullopt)
{
    return std::shared_ptr<loop_tree_program<VEX, Arithmetic>>(
        new loop_tree_program<VEX, Arithmetic>(nodes, sizes, formulas,
                                               max_interpreted_depth));
}

} // namespace loop_tree
} // namespace dabun
