// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/loop_tree/node.hpp"

namespace dabun
{
namespace loop_tree
{

template <extension VEX, class Arithmetic>
class transpose_node : public node<VEX, Arithmetic>
{

private:
    using super_type = node<VEX, Arithmetic>;

    std::string        input;
    std::string        output;
    strides_map_type   strides;
    std::optional<int> unroll_limit;

public:
    std::string dump(formulas_map_type const& /* formulas */,
                     std::map<std::string, int> const& /* sizes */,
                     std::string const& indent) const override
    {
        std::ostringstream ss;
        ss << indent << "Interpreted transpose" << std::endl;
        ss << indent << "Input: " << input << std::endl;
        ss << indent << "Output: " << output << std::endl;
        ss << utility::dump_strides(strides, indent);
        return ss.str();
    }

public:
    transpose_node(std::string const& input, std::string const& output,
                   strides_map_type const& strides,
                   std::optional<int>      unroll_limit = std::nullopt)
        : super_type(node_kind::transpose)
        , input(input)
        , output(output)
        , strides(strides)
        , unroll_limit(unroll_limit)
    {
    }

    std::string const& get_input() const { return input; }

    std::string const& get_output() const { return output; }

    std::optional<int> get_unroll_limit() const { return unroll_limit; }

    std::set<std::string> get_tensors_used() const override
    {
        return {input, output};
    }

    std::set<std::string> get_output_tensors() const override
    {
        return {output};
    }

    strides_map_type const& get_tensor_strides() const override
    {
        return strides;
    }

    std::pair<loop_tree_fn_type<Arithmetic>, report_vector>
    get_fn(std::map<std::string, int> const& tensors_idx,
           std::map<std::string, int> const&, std::map<std::string, int> const&,
           formulas_map_type const&, bool) const override
    {
        report_vector report = {
            std::make_shared<node_report>(transpose_node_info{})};

        return {[input = this->input, output = this->output,
                 input_idx  = tensors_idx.at(input),
                 output_idx = tensors_idx.at(output)](
                    std::vector<Arithmetic*>& tensors, std::vector<int>&)
                {
                    strong_assert(tensors[input_idx]);
                    strong_assert(tensors[output_idx]);

                    Arithmetic* A = tensors[input_idx];
                    Arithmetic* C = tensors[output_idx];
                    C[0]          = A[0];
                },
                report};
    }
};

template <extension VEX, class Arithmetic>
node_ptr<VEX, Arithmetic>
make_transpose_node(std::string const& input, std::string const& output,
                    strides_map_type const& strides,
                    std::optional<int>      unroll_limit = std::nullopt)
{
    return node_ptr<VEX, Arithmetic>(new transpose_node<VEX, Arithmetic>(
        input, output, strides, unroll_limit));
}

} // namespace loop_tree
} // namespace dabun
