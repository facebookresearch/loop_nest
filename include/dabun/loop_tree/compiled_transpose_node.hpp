// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/loop_tree/node.hpp"

namespace dabun
{
namespace loop_tree
{

template <extension VEX, class Arithmetic>
class compiled_transpose_node : public node<VEX, Arithmetic>
{
private:
    using ISA = typename extension_to_deprecated_ISA<VEX>::type;

    using super_type = node<VEX, Arithmetic>;

    std::string                              input;
    std::string                              output;
    std::vector<std::pair<std::string, int>> order;
    strides_map_type                         strides;
    std::optional<int>                       unroll_limit;

public:
    std::string dump(formulas_map_type const& /* formulas */,
                     std::map<std::string, int> const& sizes,
                     std::string const&                indent) const override
    {
        std::ostringstream ss;
        ss << indent << "AOT_tranpose" << std::endl;
        ss << utility::dump_order(order, indent);
        ss << utility::dump_sizes(sizes, indent);
        ss << indent << "Input: " << input << std::endl;
        ss << indent << "Output: " << output << std::endl;
        ss << utility::dump_strides(strides, indent);
        return ss.str();
    }

public:
    compiled_transpose_node(
        std::string const& input, std::string const& output,
        std::vector<std::pair<std::string, int>> const& order,
        strides_map_type const&                         strides,
        std::optional<int> unroll_limit = std::nullopt)
        : super_type(node_kind::compiled_transpose)
        , input(input)
        , output(output)
        , order(order)
        , strides(strides)
        , unroll_limit(unroll_limit)
    {
    }

    compiled_transpose_node(
        const compiled_transpose_node<VEX, Arithmetic>& other) = default;

    // creates initial transpose nest
    compiled_transpose_node(
        std::shared_ptr<for_loop_node<VEX, Arithmetic>> const&  for_node,
        std::shared_ptr<transpose_node<VEX, Arithmetic>> const& transpose_node)
        : compiled_transpose_node(
              transpose_node->get_input(), transpose_node->get_output(),
              {{for_node->get_var(), for_node->get_delta()}},
              transpose_node->get_tensor_strides(),
              transpose_node->get_unroll_limit())
    {
    }

    // extends the tranpose nest
    compiled_transpose_node(
        std::shared_ptr<for_loop_node<VEX, Arithmetic>> const& for_node,
        std::shared_ptr<compiled_transpose_node<VEX, Arithmetic>> const&
            transpose_compiler)
        : compiled_transpose_node(*transpose_compiler)
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
    }

    std::pair<loop_tree_fn_type<Arithmetic>, report_vector>
    get_fn(std::map<std::string, int> const& tensors_idx,
           std::map<std::string, int> const& sizes,
           std::map<std::string, int> const&,
           formulas_map_type const& /* formulas */,
           bool spit_asm) const override
    {
        auto aot_fn =
            transposer_compiler<
                (VEX == extension::avx512 ? extension::avx512_ymm : VEX),
                Arithmetic>(order, sizes, strides.at(output), strides.at(input),
                            64 /* unroll_limit */)
                .get_shared();

        // aot_fn.save_to_file("transpose.asm");

        std::string asm_dump = "n/a";

        if (spit_asm)
        {
            asm_dump = ::dabun::detail::get_temporary_file_name(".asm");
            aot_fn.save_to_file(asm_dump);
        }

        std::string extra_string =
            std::string("output_idx: ") +
            std::to_string(tensors_idx.at(output)) +
            ", input_idx: " + std::to_string(tensors_idx.at(input));

        compiled_transpose_node_info info{0, 0, extra_string};

        return {[aot_fn, output_idx = tensors_idx.at(output),
                 input_idx = tensors_idx.at(input)](
                    std::vector<Arithmetic*>& tensors, std::vector<int>&)
                { aot_fn(tensors[output_idx], tensors[input_idx]); },
                {std::make_shared<node_report>(info)}};
    }

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
};

} // namespace loop_tree
} // namespace dabun
