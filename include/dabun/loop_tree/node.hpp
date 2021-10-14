// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/code_generator/aot_fn.hpp"
#include "dabun/configuration.hpp"
#include "dabun/detail/tmp_file_name.hpp"
#include "dabun/isa.hpp"
#include "dabun/log.hpp"
#include "dabun/loop_nest.hpp"
#include "dabun/loop_tree/report.hpp"
#include "dabun/loop_tree/types.hpp"
#include "dabun/loop_tree/utility.hpp"

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

// Forward declarations

template <extension VEX, class Arithmetic>
class node;

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
                                             std::vector<int>& alpha_offsets) {
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
                    std::vector<Arithmetic*>& tensors, std::vector<int>&) {
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

template <extension VEX, class Arithmetic>
class for_loop_node : public node<VEX, Arithmetic>
{
private:
    using super_type = node<VEX, Arithmetic>;

    std::string var;
    int         delta;

    std::set<std::string> in_scope_tensor_names;
    std::set<std::string> in_scope_output_tensor_names;
    strides_map_type      in_scope_tensor_strides;

public:
    std::string dump(formulas_map_type const& /* formulas */,
                     std::map<std::string, int> const& /* sizes */,
                     std::string const& indent) const override
    {
        std::ostringstream ss;
        ss << indent << "Interpreted For Node" << std::endl;
        ss << indent << "Var=" << var << ", delta=" << delta << std::endl;
        return ss.str();
    }

private:
    void set_in_scope_tensor_info()
    {
        for (auto c : this->get_children())
        {
            auto node_tensor_names = c->get_tensors_used();
            in_scope_tensor_names.insert(node_tensor_names.begin(),
                                         node_tensor_names.end());

            auto node_output_tensor_names = c->get_output_tensors();
            in_scope_output_tensor_names.insert(
                node_output_tensor_names.begin(),
                node_output_tensor_names.end());

            auto node_tensor_strides = c->get_tensor_strides();

            in_scope_tensor_strides.insert(node_tensor_strides.begin(),
                                           node_tensor_strides.end());
        }
    }

    std::function<void(std::vector<Arithmetic*>&, int)>
    get_tensor_advancer(std::map<std::string, int> const& tensors_idx,
                        std::set<std::string> const&      tensor_names) const
    {
        std::vector<std::pair<int, std::int64_t>> to_advance;

        for (auto const& name : tensor_names)
        {
            if (in_scope_tensor_strides.at(name).count(var))
            {
                std::int64_t offset =
                    in_scope_tensor_strides.at(name).at(var) * delta;
                if (offset != 0)
                {
                    int idx = tensors_idx.at(name);
                    to_advance.push_back({idx, offset});
                }
            }
        }

        return [=](std::vector<Arithmetic*>& tensors, int delta = 1) {
            for (auto const& p : to_advance)
            {
                tensors[p.first] += p.second * delta;
            }
        };
    }

    std::function<void(std::vector<int>&, int)>
    get_alpha_offsets_adjuster(std::map<std::string, int> const& tensors_idx,
                               std::set<std::string> const& output_tensor_names,
                               formulas_map_type const&     formulas) const
    {

        std::vector<int> to_adjust;
        for (auto const& name : output_tensor_names)
        {
            if (formulas.count(name) && formulas.at(name).count(var) == 0)
            {
                // reduction variable, so adjust the tensor's alpha
                to_adjust.push_back(tensors_idx.at(name));
            }
        }

        return [=](std::vector<int>& alpha_offsets, int adjustment) {
            for (auto const& idx : to_adjust)
            {
                alpha_offsets[idx] += adjustment;
            }
        };
    }

public:
    std::string const& get_var() const { return var; }
    int                get_delta() const { return delta; }

    for_loop_node(std::string var, int delta,
                  std::vector<node_ptr<VEX, Arithmetic>> const& children)
        : super_type(node_kind::for_loop)
        , var(var)
        , delta(delta)
    {
        this->set_children(children);
        set_in_scope_tensor_info();
    }

    std::set<std::string> get_tensors_used() const override
    {
        return in_scope_tensor_names;
    }

    std::set<std::string> get_output_tensors() const override
    {
        return in_scope_output_tensor_names;
    }

    strides_map_type const& get_tensor_strides() const override
    {
        return in_scope_tensor_strides;
    }

    std::pair<loop_tree_fn_type<Arithmetic>, report_vector>
    get_fn(std::map<std::string, int> const& tensors_idx,
           std::map<std::string, int> const& sizes,
           std::map<std::string, int> const& iteration_depths,
           formulas_map_type const& formulas, bool spit_asm) const override
    {
        auto var      = this->var;
        auto delta    = this->delta;
        auto children = this->get_children();
        auto limit    = sizes.at(var);

        int full = limit / delta;
        int rest = limit % delta;

        report_vector report = {
            std::make_shared<node_report>(for_loop_node_info{
                1, 1, var, full + (rest ? 1 : 0), delta, limit})};

        std::vector<loop_tree_fn_type<Arithmetic>> full_fns, tail_fns;

        auto s           = sizes;
        auto iter_depths = iteration_depths;

        int n_iters = full + (rest ? 1 : 0) - 1;
        strong_assert(n_iters >= 0);

        iter_depths[var] += n_iters;

        for (auto c : children)
        {
            if (full)
            {
                s[var] = delta;
                auto sub =
                    c->get_fn(tensors_idx, s, iter_depths, formulas, spit_asm);
                full_fns.push_back(sub.first);
                report[0]->children.insert(report[0]->children.end(),
                                           sub.second.begin(),
                                           sub.second.end());
            }
            if (rest)
            {
                auto s = sizes;
                s[var] = rest;
                auto sub =
                    c->get_fn(tensors_idx, s, iter_depths, formulas, spit_asm);
                tail_fns.push_back(sub.first);
                report.insert(report.end(), sub.second.begin(),
                              sub.second.end());
            }
        }

        auto advancer = get_tensor_advancer(tensors_idx, get_tensors_used());
        auto alpha_offsets_adjuster = get_alpha_offsets_adjuster(
            tensors_idx, get_output_tensors(), formulas);

        LN_LOG(DEBUG) << "loop_tree: Executing interpreted for(" << var << ","
                      << delta << ")\n";

        return {[full, full_fns, advancer, alpha_offsets_adjuster,
                 tail_fns](std::vector<Arithmetic*>& tensors,
                           std::vector<int>&         alpha_offsets) {
                    for (int i = 0; i < full; ++i)
                    {
                        for (auto const& fn : full_fns)
                        {
                            fn(tensors, alpha_offsets);
                        }
                        advancer(tensors, 1);
                        alpha_offsets_adjuster(alpha_offsets, 1);
                    }

                    for (auto const& fn : tail_fns)
                    {
                        fn(tensors, alpha_offsets);
                    }

                    advancer(tensors, -full);
                    alpha_offsets_adjuster(alpha_offsets, -full);
                },
                report};
    }
};

template <extension VEX, class Arithmetic>
node_ptr<VEX, Arithmetic>
make_for_loop_node(std::string var, int delta,
                   std::vector<node_ptr<VEX, Arithmetic>> const& children)
{
    return node_ptr<VEX, Arithmetic>(
        new for_loop_node<VEX, Arithmetic>(var, delta, children));
}

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

        shared_aot_fn<void(Arithmetic*, Arithmetic const*, Arithmetic const*,
                           int)>
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
            asm_dump = ::dabun::detail::get_temporary_file_name(".asm");
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

        compiled_loop_nest_node_info info{generated.get_effective_flops() +
                                              generated.get_masked_out_flops(),
                                          generated.get_effective_flops(),
                                          asm_dump,
                                          generated.get_A_access_kind(),
                                          generated.get_B_access_kind(),
                                          generated.get_C_access_kind(),
                                          extra_string};

        if (extra_tensors.size() == 0)
        {
            return {[aot_fn, alpha, last_iteration,
                     input_idx_0 = tensors_idx.at(inputs[0]),
                     input_idx_1 = tensors_idx.at(inputs[1]),
                     output_idx  = tensors_idx.at(output)](
                        std::vector<Arithmetic*>& tensors,
                        std::vector<int>&         alpha_offsets) {
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
            auto aot_casted =
                aot_fn_cast<void(Arithmetic*, Arithmetic const*,
                                 Arithmetic const*, int, Arithmetic const*)>(
                    std::move(aot_fn));

            return {[aot_casted, alpha, last_iteration,
                     input_idx_0      = tensors_idx.at(inputs[0]),
                     input_idx_1      = tensors_idx.at(inputs[1]),
                     output_idx       = tensors_idx.at(output),
                     extra_tensor_idx = tensors_idx.at(extra_tensors[0])](
                        std::vector<Arithmetic*>& tensors,
                        std::vector<int>&         alpha_offsets) {
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
            auto aot_casted = aot_fn_cast<void(
                Arithmetic*, Arithmetic const*, Arithmetic const*, int,
                Arithmetic const*, Arithmetic const*)>(std::move(aot_fn));

            return {[aot_casted, alpha, last_iteration,
                     input_idx_0        = tensors_idx.at(inputs[0]),
                     input_idx_1        = tensors_idx.at(inputs[1]),
                     output_idx         = tensors_idx.at(output),
                     extra_tensor_idx_0 = tensors_idx.at(extra_tensors[0]),
                     extra_tensor_idx_1 = tensors_idx.at(extra_tensors[1])](
                        std::vector<Arithmetic*>& tensors,
                        std::vector<int>&         alpha_offsets) {
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
                    std::vector<Arithmetic*>& tensors, std::vector<int>&) {
                    aot_fn(tensors[output_idx], tensors[input_idx]);
                },
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
