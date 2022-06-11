// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/loop_tree/node.hpp"
#include "dabun/utility/tmp_file_name.hpp"

namespace dabun
{
namespace loop_tree
{

template <extension VEX, class Arithmetic>
class compiled_loop_nest_node : public node<VEX, Arithmetic>
{
private:
    using ISA = typename extension_to_deprecated_ISA<VEX>::type;

    using super_type = node<VEX, Arithmetic>;

    std::vector<std::string>                 inputs;
    std::string                              output;
    std::vector<std::pair<std::string, int>> order;
    strides_map_type                         strides;
    arithmetic_op_kind                       plus;
    arithmetic_op_kind                       multiplies;
    int                                      alpha;

    std::optional<int>                       unroll_limit;
    elementwise_op_ptr<ISA>                  elementwise_preop;
    std::vector<std::string>                 elementwise_preop_tensors;
    elementwise_op_ptr<ISA>                  elementwise_postop;
    std::vector<std::string>                 elementwise_postop_tensors;
    std::optional<OptimizationConfiguration> optim_config;

public:
    std::string dump(formulas_map_type const&          formulas,
                     std::map<std::string, int> const& sizes,
                     std::string const&                indent) const override
    {
        std::ostringstream ss;
        ss << indent << "AOT_loop_nest" << std::endl;
        ss << utility::dump_order(order, indent);

        ss << utility::dump_sizes(sizes, indent);

        ss << indent << "Inputs: ";
        ss << utility::dump_tensors(inputs, indent);

        ss << indent << "Output: " << output << std::endl;

        if (elementwise_preop_tensors.size())
        {
            ss << indent << "Preop Tensors: ";
            ss << utility::dump_tensors(elementwise_preop_tensors, indent);
        }

        if (elementwise_postop_tensors.size())
        {
            ss << indent << "Postop Tensors: ";
            ss << utility::dump_tensors(elementwise_postop_tensors, indent);
        }

        ss << utility::dump_strides(strides, indent);
        ss << utility::dump_formula(formulas, indent);
        return ss.str();
    }

public:
    compiled_loop_nest_node(
        std::vector<std::string> const& inputs, std::string const& output,
        std::vector<std::pair<std::string, int>> const& order,
        strides_map_type const& strides, arithmetic_op_kind plus,
        arithmetic_op_kind multiplies, int alpha,
        std::optional<int>              unroll_limit      = std::nullopt,
        elementwise_op_ptr<ISA> const&  elementwise_preop = nullptr,
        std::vector<std::string> const& elementwise_preop_tensors  = {},
        elementwise_op_ptr<ISA> const&  elementwise_postop         = nullptr,
        std::vector<std::string> const& elementwise_postop_tensors = {},
        std::optional<OptimizationConfiguration> optim_config = std::nullopt)
        : super_type(node_kind::compiled_loop_nest)
        , inputs(inputs)
        , output(output)
        , order(order)
        , strides(strides)
        , plus(plus)
        , multiplies(multiplies)
        , alpha(alpha)
        , unroll_limit(unroll_limit)
        , elementwise_preop(elementwise_preop)
        , elementwise_preop_tensors(elementwise_preop_tensors)
        , elementwise_postop(elementwise_postop)
        , elementwise_postop_tensors(elementwise_postop_tensors)
        , optim_config(optim_config)
    {
    }

    compiled_loop_nest_node(
        compiled_loop_nest_node<VEX, Arithmetic> const& other) = default;

    // creates an initial loop nest
    compiled_loop_nest_node(
        std::shared_ptr<for_loop_node<VEX, Arithmetic>> const& for_node,
        std::shared_ptr<compute_node<VEX, Arithmetic>> const&  compute_node)
        : compiled_loop_nest_node(
              compute_node->get_inputs(), compute_node->get_output(),
              {{for_node->get_var(), for_node->get_delta()}},
              compute_node->get_tensor_strides(), compute_node->get_plus(),
              compute_node->get_multiplies(), compute_node->get_alpha(),
              compute_node->get_unroll_limit(),
              compute_node->get_elementwise_preop(),
              compute_node->get_elementwise_preop_tensors(),
              compute_node->get_elementwise_postop(),
              compute_node->get_elementwise_postop_tensors(),
              compute_node->get_optim_config())
    {
    }

    // extends an existing loop nest
    compiled_loop_nest_node(
        std::shared_ptr<for_loop_node<VEX, Arithmetic>> const& for_node,
        std::shared_ptr<compiled_loop_nest_node<VEX, Arithmetic>> const&
            compute_compiler_node)
        : compiled_loop_nest_node(*compute_compiler_node)
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
        // TODO(zi): are we missing the corresponding strides here?
    }

    std::pair<loop_tree_fn_type<Arithmetic>, report_vector>
    get_fn(std::map<std::string, int> const& tensors_idx,
           std::map<std::string, int> const& sizes,
           std::map<std::string, int> const& iteration_depths,
           formulas_map_type const& formulas, bool spit_asm) const override
    {
        // contains followed tensors for pre/post ops
        std::vector<std::string> extra_tensors;

        std::vector<std::map<std::string, int>> preop_strides;
        for (auto const& name : elementwise_preop_tensors)
        {
            preop_strides.push_back(strides.at(name));
            extra_tensors.push_back(name);
        }

        std::vector<std::map<std::string, int>> postop_strides;
        for (auto const& name : elementwise_postop_tensors)
        {
            postop_strides.push_back(strides.at(name));
            extra_tensors.push_back(name);
        }

#ifdef SERIALIZE_LOOP_NEST
        save_loop_nest_inputs(DABUN_STRINGIFY(SERIALIZE_LOOP_NEST), order,
                              sizes, formulas.at(output),
                              formulas.at(inputs[0]), formulas.at(inputs[1]),
                              strides.at(output), strides.at(inputs[0]),
                              strides.at(inputs[1]), unroll_limit);
#endif

        shared_code_generated_fn<void(Arithmetic*, Arithmetic const*,
                                      Arithmetic const*, int)>
            aot_fn;

        loop_nest_compiler<VEX, Arithmetic> generated(
            order, sizes, formulas.at(output), formulas.at(inputs[0]),
            formulas.at(inputs[1]), strides.at(output), strides.at(inputs[0]),
            strides.at(inputs[1]),
            utility::get_operation_pair(plus, multiplies), unroll_limit,
            elementwise_preop, preop_strides, elementwise_postop,
            postop_strides, optim_config);

        std::string asm_dump = "n/a";

        if (spit_asm)
        {
            asm_dump = ::dabun::utility::get_temporary_file_name(".asm");
        }

        aot_fn = std::move(generated).get_shared();
        // aot_fn.save_to_file("loop_nest.asm");

        if (spit_asm)
        {
            aot_fn.save_to_file(asm_dump);
        }

        auto output = this->output;
        auto inputs = this->inputs;
        auto alpha  = this->alpha;

        int         last_iteration  = 0;
        auto const& output_strides  = strides.at(output);
        auto const& input_0_strides = strides.at(inputs.at(0));
        auto const& input_1_strides = strides.at(inputs.at(1));

        for (auto const& p : iteration_depths)
        {
            if (output_strides.count(p.first) == 0 &&
                (input_0_strides.count(p.first) &&
                 input_1_strides.count(p.first)))
            {
                last_iteration += p.second;
            }
        }

        std::string extra_string =
            std::string("alpha: ") + std::to_string(alpha) +
            ", last_iteration: " + std::to_string(last_iteration) +
            ", input_idx_0: " + std::to_string(tensors_idx.at(inputs[0])) +
            ", input_idx_1: " + std::to_string(tensors_idx.at(inputs[1])) +
            ", outut_idx: " + std::to_string(tensors_idx.at(output));

        compiled_loop_nest_node_info info{
            generated.get_effective_flops() + generated.get_masked_out_flops(),
            generated.get_effective_flops(),
            asm_dump,
            generated.get_A_access_kind(),
            generated.get_B_access_kind(),
            generated.get_C_access_kind(),
            generated.get_register_blocking_info(),
            extra_string};

        if (extra_tensors.size() == 0)
        {
            return {[aot_fn, alpha, last_iteration,
                     input_idx_0 = tensors_idx.at(inputs[0]),
                     input_idx_1 = tensors_idx.at(inputs[1]),
                     output_idx  = tensors_idx.at(output)](
                        std::vector<Arithmetic*>& tensors,
                        std::vector<int>&         alpha_offsets)
                    {
                        auto last_iter_mask =
                            alpha_offsets[output_idx] == last_iteration ? 0b0
                                                                        : 0b10;
                        auto param_mask =
                            ((alpha | alpha_offsets[output_idx]) ? 1 : 0) |
                            last_iter_mask;

                        aot_fn(tensors[output_idx], tensors[input_idx_0],
                               tensors[input_idx_1], param_mask);
                    },
                    {std::make_shared<node_report>(info)}};
        }
        else if (extra_tensors.size() == 1)
        {
            auto aot_casted = code_generated_fn_cast<void(
                Arithmetic*, Arithmetic const*, Arithmetic const*, int,
                Arithmetic const*)>(std::move(aot_fn));

            return {[aot_casted, alpha, last_iteration,
                     input_idx_0      = tensors_idx.at(inputs[0]),
                     input_idx_1      = tensors_idx.at(inputs[1]),
                     output_idx       = tensors_idx.at(output),
                     extra_tensor_idx = tensors_idx.at(extra_tensors[0])](
                        std::vector<Arithmetic*>& tensors,
                        std::vector<int>&         alpha_offsets)
                    {
                        aot_casted(
                            tensors[output_idx], tensors[input_idx_0],
                            tensors[input_idx_1],
                            ((alpha + alpha_offsets[output_idx]) ? 1 : 0) |
                                (alpha_offsets[output_idx] == last_iteration
                                     ? 0
                                     : 2),
                            tensors[extra_tensor_idx]);
                    },
                    {std::make_shared<node_report>(info)}};
        }
        else if (extra_tensors.size() == 2)
        {
            auto aot_casted = code_generated_fn_cast<void(
                Arithmetic*, Arithmetic const*, Arithmetic const*, int,
                Arithmetic const*, Arithmetic const*)>(std::move(aot_fn));

            return {[aot_casted, alpha, last_iteration,
                     input_idx_0        = tensors_idx.at(inputs[0]),
                     input_idx_1        = tensors_idx.at(inputs[1]),
                     output_idx         = tensors_idx.at(output),
                     extra_tensor_idx_0 = tensors_idx.at(extra_tensors[0]),
                     extra_tensor_idx_1 = tensors_idx.at(extra_tensors[1])](
                        std::vector<Arithmetic*>& tensors,
                        std::vector<int>&         alpha_offsets)
                    {
                        aot_casted(
                            tensors[output_idx], tensors[input_idx_0],
                            tensors[input_idx_1],
                            ((alpha + alpha_offsets[output_idx]) ? 1 : 0) |
                                (alpha_offsets[output_idx] == last_iteration
                                     ? 0
                                     : 2),
                            tensors[extra_tensor_idx_0],
                            tensors[extra_tensor_idx_1]);
                    },
                    {std::make_shared<node_report>(info)}};
        }
        else
        {
            throw std::invalid_argument(
                "loop_nest currently supports at most 2 followed tensors");
            return {loop_tree_fn_type<Arithmetic>(), report_vector{}};
        }
    }

    std::set<std::string> get_tensors_used() const override
    {
        std::set<std::string> tensors_used(inputs.begin(), inputs.end());
        tensors_used.insert(output);
        return tensors_used;
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
