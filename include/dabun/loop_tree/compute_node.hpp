// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/loop_tree/node.hpp"

namespace dabun
{
namespace loop_tree
{

template <extension VEX, class Arithmetic>
class compute_node : public node<VEX, Arithmetic>
{
private:
    using ISA = typename extension_to_deprecated_ISA<VEX>::type;

    using super_type = node<VEX, Arithmetic>;

    // 0 -> A, 1 -> B, rest are followed tensors
    std::vector<std::string> inputs;
    std::string              output;
    strides_map_type         strides;
    arithmetic_op_kind       plus;
    arithmetic_op_kind       multiplies;
    int                      alpha;

    std::optional<int>                       unroll_limit;
    elementwise_op_ptr<ISA>                  elementwise_preop;
    std::vector<std::string>                 elementwise_preop_tensors;
    elementwise_op_ptr<ISA>                  elementwise_postop;
    std::vector<std::string>                 elementwise_postop_tensors;
    std::optional<OptimizationConfiguration> optim_config;

public:
    std::string dump(formulas_map_type const& formulas,
                     std::map<std::string, int> const& /* sizes */,
                     std::string const& indent) const override
    {
        std::ostringstream ss;
        ss << indent << "Interpreted Compute Node" << std::endl;

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
    compute_node(
        std::vector<std::string> const& inputs, std::string const& output,
        strides_map_type const& strides, arithmetic_op_kind plus,
        arithmetic_op_kind multiplies, int alpha,
        std::optional<int>              unroll_limit      = std::nullopt,
        elementwise_op_ptr<ISA> const&  elementwise_preop = nullptr,
        std::vector<std::string> const& elementwise_preop_tensors  = {},
        elementwise_op_ptr<ISA> const&  elementwise_postop         = nullptr,
        std::vector<std::string> const& elementwise_postop_tensors = {},
        std::optional<OptimizationConfiguration> optim_config = std::nullopt)
        : super_type(node_kind::compute)
        , inputs(inputs)
        , output(output)
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
        for (auto const& t : inputs)
        {
            strong_assert(strides.count(t) > 0);
        }

        strong_assert(strides.count(output));

        if (!elementwise_preop_tensors.empty())
        {
            strong_assert(elementwise_preop != nullptr);
        }

        for (auto const& t : elementwise_preop_tensors)
        {
            strong_assert(strides.count(t) > 0);
        }

        if (!elementwise_postop_tensors.empty())
        {
            strong_assert(elementwise_postop != nullptr);
        }

        for (auto const& t : elementwise_postop_tensors)
        {
            strong_assert(strides.count(t) > 0);
        }
    }

    std::string const& get_output() const { return output; }

    std::vector<std::string> const& get_inputs() const { return inputs; }

    arithmetic_op_kind get_plus() const { return plus; }

    arithmetic_op_kind get_multiplies() const { return multiplies; }

    int get_alpha() const { return alpha; }

    std::optional<int> get_unroll_limit() const { return unroll_limit; }

    elementwise_op_ptr<ISA> get_elementwise_preop() const
    {
        return elementwise_preop;
    }

    std::vector<std::string> const& get_elementwise_preop_tensors() const
    {
        return elementwise_preop_tensors;
    }

    elementwise_op_ptr<ISA> get_elementwise_postop() const
    {
        return elementwise_postop;
    }

    std::vector<std::string> const& get_elementwise_postop_tensors() const
    {
        return elementwise_postop_tensors;
    }

    std::optional<OptimizationConfiguration> get_optim_config() const
    {
        return optim_config;
    }

    strides_map_type const& get_tensor_strides() const override
    {
        return strides;
    }

    std::set<std::string> get_tensors_used() const override
    {
        std::set<std::string> tensors_used(inputs.begin(), inputs.end());
        tensors_used.insert(output);

        tensors_used.insert(elementwise_preop_tensors.begin(),
                            elementwise_preop_tensors.end());
        tensors_used.insert(elementwise_postop_tensors.begin(),
                            elementwise_postop_tensors.end());

        return tensors_used;
    }

    std::set<std::string> get_output_tensors() const override
    {
        return {output};
    }

    std::pair<loop_tree_fn_type<Arithmetic>, report_vector>
    get_fn(std::map<std::string, int> const& tensors_idx,
           std::map<std::string, int> const& /* sizes */,
           std::map<std::string, int> const& /* iteration_depths */,
           formulas_map_type const&, bool) const override
    {
        // TODO(j): if we want to support more ops, extend here otherwise only
        // supported through loop nest
        if (plus != arithmetic_op_kind::plus ||
            multiplies != arithmetic_op_kind::multiplies)
        {
            throw std::invalid_argument("Interpreted compute only supports "
                                        "standard plus and multiplies");
        }

        if (!elementwise_preop_tensors.empty() ||
            !elementwise_postop_tensors.empty())
        {
            throw std::invalid_argument("Interpreted compute doesn't support "
                                        "pre/post op with followed tensors");
        }

        report_vector report = {
            std::make_shared<node_report>(compute_node_info{})};

        return {[alpha = this->alpha, input_idx_0 = tensors_idx.at(inputs[0]),
                 input_idx_1 = tensors_idx.at(inputs[1]),
                 output_idx =
                     tensors_idx.at(output)](std::vector<Arithmetic*>& tensors,
                                             std::vector<int>& alpha_offsets)
                {
                    Arithmetic* A = tensors[input_idx_0];
                    Arithmetic* B = tensors[input_idx_1];
                    Arithmetic* C = tensors[output_idx];
                    if ((alpha + alpha_offsets[output_idx]) == 0)
                    {
                        C[0] = 0.0;
                    }

                    C[0] += A[0] * B[0];
                },
                report};
    }
};

template <extension VEX, class Arithmetic>
node_ptr<VEX, Arithmetic> make_compute_node(
    std::vector<std::string> const& inputs, std::string const& output,
    strides_map_type const& strides, arithmetic_op_kind plus,
    arithmetic_op_kind multiplies, int alpha,
    std::optional<int> unroll_limit = std::nullopt,
    elementwise_op_ptr<extension_to_deprecated_ISA_t<VEX>> const&
                                    elementwise_preop         = nullptr,
    std::vector<std::string> const& elementwise_preop_tensors = {},
    elementwise_op_ptr<extension_to_deprecated_ISA_t<VEX>> const&
                                             elementwise_postop = nullptr,
    std::vector<std::string> const&          elementwise_postop_tensors = {},
    std::optional<OptimizationConfiguration> optim_config = std::nullopt)
{
    return node_ptr<VEX, Arithmetic>(new compute_node<VEX, Arithmetic>(
        inputs, output, strides, plus, multiplies, alpha, unroll_limit,
        elementwise_preop, elementwise_preop_tensors, elementwise_postop,
        elementwise_postop_tensors, optim_config));
}

} // namespace loop_tree
} // namespace dabun
