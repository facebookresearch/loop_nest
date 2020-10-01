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
#include "dabun/loop_tree/types.hpp"
#include "dabun/serialization.hpp"
#include "dabun/transposer.hpp"

namespace dabun
{
namespace loop_tree
{

enum class node_kind
{
    for_loop,
    compute,
    transpose,
    jitted_loop_nest,
    jitted_transpose
};

inline std::map<node_kind, std::string> node_kind_to_str = {
    {node_kind::for_loop, "for_loop_node"},
    {node_kind::compute, "compute_node"},
    {node_kind::transpose, "transpose_node"},
    {node_kind::jitted_loop_nest, "jitted_loop_nest_node"},
    {node_kind::jitted_transpose, "jitted_transpose_node"}};

inline std::string get_node_kind_str(node_kind kind)
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

inline std::shared_ptr<operation_pair_base>
get_operation_pair(arithmetic_op_kind plus_op, arithmetic_op_kind multiplies_op)
{

    std::map<std::pair<arithmetic_op_kind, arithmetic_op_kind>,
             std::shared_ptr<operation_pair_base>>
#ifndef LOOP_NEST_ARM
        op_map = {
            {{arithmetic_op_kind::plus, arithmetic_op_kind::multiplies},
             std::make_shared<
                 operation_pair<op::basic_plus, op::basic_multiplies>>()},
            {{arithmetic_op_kind::max, arithmetic_op_kind::multiplies},
             std::make_shared<operation_pair<op::max, op::basic_multiplies>>()},
            {{arithmetic_op_kind::min, arithmetic_op_kind::multiplies},
             std::make_shared<operation_pair<op::min, op::basic_multiplies>>()},
            {{arithmetic_op_kind::max, arithmetic_op_kind::plus},
             std::make_shared<operation_pair<op::max, op::basic_plus>>()}};
#else
        op_map = {{{arithmetic_op_kind::plus, arithmetic_op_kind::multiplies},
                   std::make_shared<operation_pair_base>()}};
#endif

    return op_map.at({plus_op, multiplies_op});
}

template <class ISA>
class loop_tree_node;

template <class ISA>
class compute_node;

template <class ISA>
class jitted_loop_nest_node;

template <class ISA>
class transpose_node;

template <class ISA>
class jitted_transpose_node;

template <class ISA>
class for_loop_node;

inline std::string dump_strides(strides_map_type const& strides,
                                std::string const&      indent)
{
    std::ostringstream ss;
    ss << indent << "Strides: " << std::endl;
    for (auto const& tensor_strides : strides)
    {
        // tensor
        ss << indent << " " << tensor_strides.first << ": ";
        // strides
        for (auto const& entry : tensor_strides.second)
        {
            ss << entry.first << ":" << entry.second << " ";
        }
        ss << std::endl;
    }
    return ss.str();
}

inline std::string dump_formula(formulas_map_type const& formulas,
                                std::string const&       indent)
{
    std::ostringstream ss;
    ss << indent << "Formulas: " << std::endl;
    for (auto const& tensor_formula : formulas)
    {
        // tensor
        ss << indent << " " << tensor_formula.first << ": ";
        // formula
        for (auto const& entry : tensor_formula.second)
        {
            ss << entry << " ";
        }
        ss << std::endl;
    }
    return ss.str();
}

inline std::string dump_tensors(std::vector<std::string> const& tensors,
                                std::string const&              indent)
{
    std::ostringstream ss;
    for (auto const& i : tensors)
    {
        ss << i << " ";
    }
    ss << std::endl;
    return ss.str();
}

inline std::string
dump_order(std::vector<std::pair<std::string, int>> const& order,
           std::string const&                              indent)
{
    std::ostringstream ss;
    ss << indent << "Order: ";
    for (auto const& o : order)
    {
        ss << o.first << ":" << o.second << " ";
    }
    ss << std::endl;
    return ss.str();
}

inline std::string dump_sizes(std::map<std::string, int> const& sizes,
                              std::string const&                indent)
{
    std::ostringstream ss;
    ss << indent << "Sizes: ";
    for (auto const& s : sizes)
    {
        ss << s.first << ":" << s.second << " ";
    }
    ss << std::endl;
    return ss.str();
}

template <class ISA>
class loop_tree_node
{

private:
    node_kind                            kind_;
    std::vector<loop_tree_node_ptr<ISA>> children_;

public:
    virtual ~loop_tree_node(){};

    explicit loop_tree_node(node_kind kind)
        : kind_(kind)
    {
    }

    std::vector<loop_tree_node_ptr<ISA>> const& get_children() const
    {
        return children_;
    }

    void set_children(std::vector<loop_tree_node_ptr<ISA>> const& children)
    {
        children_ = children;
    }

    void set_children(std::vector<loop_tree_node_ptr<ISA>>&& children)
    {
        children_ = std::move(children);
    }

    node_kind get_type() const { return kind_; }

    // tensor positions, dimension sizes, and tensor formulas
    virtual loop_tree_fn_type get_fn(std::map<std::string, int> const&,
                                     std::map<std::string, int> const&,
                                     formulas_map_type const&) const = 0;

    virtual std::set<std::string> get_tensors_used() const = 0;

    virtual std::set<std::string> get_output_tensors() const = 0;

    virtual strides_map_type const& get_tensor_strides() const = 0;

    virtual std::string dump(formulas_map_type const&          formulas,
                             std::map<std::string, int> const& sizes,
                             std::string const& indent) const = 0;
};

template <class ISA>
class compute_node : public loop_tree_node<ISA>
{
private:
    using super_type = loop_tree_node<ISA>;

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
    std::string dump(formulas_map_type const&          formulas,
                     std::map<std::string, int> const& sizes,
                     std::string const&                indent) const override
    {
        std::ostringstream ss;
        ss << indent << "Interpreted Compute Node" << std::endl;

        ss << indent << "Inputs: ";
        ss << dump_tensors(inputs, indent);

        ss << indent << "Output: " << output << std::endl;

        if (elementwise_preop_tensors.size())
        {
            ss << indent << "Preop Tensors: ";
            ss << dump_tensors(elementwise_preop_tensors, indent);
        }

        if (elementwise_postop_tensors.size())
        {
            ss << indent << "Postop Tensors: ";
            ss << dump_tensors(elementwise_postop_tensors, indent);
        }

        ss << dump_strides(strides, indent);
        ss << dump_formula(formulas, indent);
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

    loop_tree_fn_type get_fn(std::map<std::string, int> const& tensors_idx,
                             std::map<std::string, int> const& sizes,
                             formulas_map_type const&) const override
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

        return [inputs = this->inputs, output = this->output,
                alpha = this->alpha, input_idx_0 = tensors_idx.at(inputs[0]),
                input_idx_1 = tensors_idx.at(inputs[1]),
                output_idx  = tensors_idx.at(output)](
                   std::vector<float*> const& tensors,
                   std::vector<int> const&    alpha_offsets) {
            float* A = tensors[input_idx_0];
            float* B = tensors[input_idx_1];
            float* C = tensors[output_idx];
            if ((alpha + alpha_offsets[output_idx]) == 0)
            {
                C[0] = 0.0;
            }

            C[0] += A[0] * B[0];
        };
    }
};

template <class ISA>
loop_tree_node_ptr<ISA> make_compute_node(
    std::vector<std::string> const& inputs, std::string const& output,
    strides_map_type const& strides, arithmetic_op_kind plus,
    arithmetic_op_kind multiplies, int alpha,
    std::optional<int>                       unroll_limit      = std::nullopt,
    elementwise_op_ptr<ISA> const&           elementwise_preop = nullptr,
    std::vector<std::string> const&          elementwise_preop_tensors = {},
    elementwise_op_ptr<ISA> const&           elementwise_postop = nullptr,
    std::vector<std::string> const&          elementwise_postop_tensors = {},
    std::optional<OptimizationConfiguration> optim_config = std::nullopt)
{
    return loop_tree_node_ptr<ISA>(new compute_node<ISA>(
        inputs, output, strides, plus, multiplies, alpha, unroll_limit,
        elementwise_preop, elementwise_preop_tensors, elementwise_postop,
        elementwise_postop_tensors, optim_config));
}

template <class ISA>
class transpose_node : public loop_tree_node<ISA>
{

private:
    using super_type = loop_tree_node<ISA>;

    std::string        input;
    std::string        output;
    strides_map_type   strides;
    std::optional<int> unroll_limit;

public:
    std::string dump(formulas_map_type const&          formulas,
                     std::map<std::string, int> const& sizes,
                     std::string const&                indent) const override
    {
        std::ostringstream ss;
        ss << indent << "Interpreted transpose" << std::endl;
        ss << indent << "Input: " << input << std::endl;
        ss << indent << "Output: " << output << std::endl;
        ss << dump_strides(strides, indent);
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

    loop_tree_fn_type get_fn(std::map<std::string, int> const& tensors_idx,
                             std::map<std::string, int> const&,
                             formulas_map_type const&) const override
    {
        return
            [input = this->input, output = this->output,
             input_idx  = tensors_idx.at(input),
             output_idx = tensors_idx.at(output)](
                std::vector<float*> const& tensors, std::vector<int> const&) {
                strong_assert(tensors[input_idx]);
                strong_assert(tensors[output_idx]);

                float* A = tensors[input_idx];
                float* C = tensors[output_idx];
                C[0]     = A[0];
            };
    }
}; // namespace aot

template <class ISA>
loop_tree_node_ptr<ISA>
make_transpose_node(std::string const& input, std::string const& output,
                    strides_map_type const& strides,
                    std::optional<int>      unroll_limit = std::nullopt)
{
    return loop_tree_node_ptr<ISA>(
        new transpose_node<ISA>(input, output, strides, unroll_limit));
}

template <class ISA>
class for_loop_node : public loop_tree_node<ISA>
{
private:
    using super_type = loop_tree_node<ISA>;

    std::string var;
    int         delta;

    std::set<std::string> in_scope_tensor_names;
    std::set<std::string> in_scope_output_tensor_names;
    strides_map_type      in_scope_tensor_strides;

public:
    std::string dump(formulas_map_type const&          formulas,
                     std::map<std::string, int> const& sizes,
                     std::string const&                indent) const override
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

    std::function<void(std::vector<float*>&)>
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
                int idx = tensors_idx.at(name);
                to_advance.push_back({idx, offset});
            }
        }

        return [=](std::vector<float*>& tensors) {
            for (auto const& p : to_advance)
            {
                tensors[p.first] += p.second;
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
                  std::vector<loop_tree_node_ptr<ISA>> const& children)
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

    loop_tree_fn_type get_fn(std::map<std::string, int> const& tensors_idx,
                             std::map<std::string, int> const& sizes,
                             formulas_map_type const& formulas) const override
    {
        auto var      = this->var;
        auto delta    = this->delta;
        auto children = this->get_children();
        auto limit    = sizes.at(var);

        int full = limit / delta;
        int rest = limit % delta;

        std::vector<loop_tree_fn_type> full_fns, tail_fns;

        for (auto c : children)
        {
            if (full)
            {
                auto s = sizes;
                s[var] = delta;
                full_fns.push_back(c->get_fn(tensors_idx, s, formulas));
            }
            if (rest)
            {
                auto s = sizes;
                s[var] = rest;
                tail_fns.push_back(c->get_fn(tensors_idx, s, formulas));
            }
        }

        auto advancer = get_tensor_advancer(tensors_idx, get_tensors_used());
        auto alpha_offsets_adjuster = get_alpha_offsets_adjuster(
            tensors_idx, get_output_tensors(), formulas);

        LN_LOG(DEBUG) << "loop_tree: Executing interpreted for(" << var << ","
                      << delta << ")\n";

        return
            [=](std::vector<float*> tensors, std::vector<int> alpha_offsets) {
                for (int i = 0; i < full; ++i)
                {
                    for (auto const& fn : full_fns)
                    {
                        fn(tensors, alpha_offsets);
                    }
                    advancer(tensors);
                    alpha_offsets_adjuster(alpha_offsets, 2);
                }

                for (auto const& fn : tail_fns)
                {
                    fn(tensors, alpha_offsets);
                }
            };
    }
};

template <class ISA>
loop_tree_node_ptr<ISA>
make_for_loop_node(std::string var, int delta,
                   std::vector<loop_tree_node_ptr<ISA>> const& children)
{
    return loop_tree_node_ptr<ISA>(
        new for_loop_node<ISA>(var, delta, children));
}

template <class ISA>
class jitted_loop_nest_node : public loop_tree_node<ISA>
{
private:
    using super_type = loop_tree_node<ISA>;

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
        ss << indent << "JIT_loop_nest" << std::endl;
        ss << dump_order(order, indent);

        ss << dump_sizes(sizes, indent);

        ss << indent << "Inputs: ";
        ss << dump_tensors(inputs, indent);

        ss << indent << "Output: " << output << std::endl;

        if (elementwise_preop_tensors.size())
        {
            ss << indent << "Preop Tensors: ";
            ss << dump_tensors(elementwise_preop_tensors, indent);
        }

        if (elementwise_postop_tensors.size())
        {
            ss << indent << "Postop Tensors: ";
            ss << dump_tensors(elementwise_postop_tensors, indent);
        }

        ss << dump_strides(strides, indent);
        ss << dump_formula(formulas, indent);
        return ss.str();
    }

public:
    jitted_loop_nest_node(
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
        : super_type(node_kind::jitted_loop_nest)
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

    jitted_loop_nest_node(jitted_loop_nest_node<ISA> const& other) = default;

    // creates an initial loop nest
    jitted_loop_nest_node(
        std::shared_ptr<for_loop_node<ISA>> const& for_node,
        std::shared_ptr<compute_node<ISA>> const&  compute_node)
        : jitted_loop_nest_node(
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
    jitted_loop_nest_node(
        std::shared_ptr<for_loop_node<ISA>> const&         for_node,
        std::shared_ptr<jitted_loop_nest_node<ISA>> const& compute_jitter_node)
        : jitted_loop_nest_node(*compute_jitter_node)
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
    }

    loop_tree_fn_type get_fn(std::map<std::string, int> const& tensors_idx,
                             std::map<std::string, int> const& sizes,
                             formulas_map_type const& formulas) const override
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

        auto jit_fn =
            loop_nest_code_generator<ISA>(
                order, sizes, formulas.at(output), formulas.at(inputs[0]),
                formulas.at(inputs[1]), strides.at(output),
                strides.at(inputs[0]), strides.at(inputs[1]),
                get_operation_pair(plus, multiplies), unroll_limit,
                elementwise_preop, preop_strides, elementwise_postop,
                postop_strides, optim_config)
                .get_shared();

        auto output = this->output;
        auto inputs = this->inputs;
        auto alpha  = this->alpha;

        if (extra_tensors.size() == 0)
        {
            return [jit_fn, alpha, input_idx_0 = tensors_idx.at(inputs[0]),
                    input_idx_1 = tensors_idx.at(inputs[1]),
                    output_idx  = tensors_idx.at(output)](
                       std::vector<float*> const& tensors,
                       std::vector<int> const&    alpha_offsets) {
                jit_fn(tensors[output_idx], tensors[input_idx_0],
                       tensors[input_idx_1],
                       alpha + alpha_offsets[output_idx] ? 1 : 0);
            };
        }
        else if (extra_tensors.size() == 1)
        {
            auto jit_fn_cast =
                aot_fn_cast<void(float*, float const*, float const*, int,
                                 float const*)>(std::move(jit_fn));

            return [jit_fn_cast, alpha, input_idx_0 = tensors_idx.at(inputs[0]),
                    input_idx_1      = tensors_idx.at(inputs[1]),
                    output_idx       = tensors_idx.at(output),
                    extra_tensor_idx = tensors_idx.at(extra_tensors[0])](
                       std::vector<float*> const& tensors,
                       std::vector<int> const&    alpha_offsets) {
                jit_fn_cast(tensors[output_idx], tensors[input_idx_0],
                            tensors[input_idx_1],
                            alpha + alpha_offsets[output_idx] ? 1 : 0,
                            tensors[extra_tensor_idx]);
            };
        }
        else if (extra_tensors.size() == 2)
        {
            auto jit_fn_cast =
                aot_fn_cast<void(float*, float const*, float const*, int,
                                 float const*, float const*)>(
                    std::move(jit_fn));

            return [jit_fn_cast, alpha, input_idx_0 = tensors_idx.at(inputs[0]),
                    input_idx_1        = tensors_idx.at(inputs[1]),
                    output_idx         = tensors_idx.at(output),
                    extra_tensor_idx_0 = tensors_idx.at(extra_tensors[0]),
                    extra_tensor_idx_1 = tensors_idx.at(extra_tensors[1])](
                       std::vector<float*> const& tensors,
                       std::vector<int> const&    alpha_offsets) {
                jit_fn_cast(tensors[output_idx], tensors[input_idx_0],
                            tensors[input_idx_1],
                            alpha + alpha_offsets[output_idx] ? 1 : 0,
                            tensors[extra_tensor_idx_0],
                            tensors[extra_tensor_idx_1]);
            };
        }
        else
        {
            throw std::invalid_argument(
                "loop_nest currently supports at most 2 followed tensors");
            return loop_tree_fn_type();
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

template <class ISA>
class jitted_transpose_node : public loop_tree_node<ISA>
{
private:
    using super_type = loop_tree_node<ISA>;

    std::string                              input;
    std::string                              output;
    std::vector<std::pair<std::string, int>> order;
    strides_map_type                         strides;
    std::optional<int>                       unroll_limit;

public:
    std::string dump(formulas_map_type const&          formulas,
                     std::map<std::string, int> const& sizes,
                     std::string const&                indent) const override
    {
        std::ostringstream ss;
        ss << indent << "JIT_tranpose" << std::endl;
        ss << dump_order(order, indent);
        ss << dump_sizes(sizes, indent);
        ss << indent << "Input: " << input << std::endl;
        ss << indent << "Output: " << output << std::endl;
        ss << dump_strides(strides, indent);
        return ss.str();
    }

public:
    jitted_transpose_node(std::string const& input, std::string const& output,
                          std::vector<std::pair<std::string, int>> const& order,
                          strides_map_type const& strides,
                          std::optional<int>      unroll_limit = std::nullopt)
        : super_type(node_kind::jitted_transpose)
        , input(input)
        , output(output)
        , order(order)
        , strides(strides)
        , unroll_limit(unroll_limit)
    {
    }

    jitted_transpose_node(const jitted_transpose_node<ISA>& other) = default;

    // creates initial transpose nest
    jitted_transpose_node(
        std::shared_ptr<for_loop_node<ISA>> const&  for_node,
        std::shared_ptr<transpose_node<ISA>> const& transpose_node)
        : jitted_transpose_node(transpose_node->get_input(),
                                transpose_node->get_output(),
                                {{for_node->get_var(), for_node->get_delta()}},
                                transpose_node->get_tensor_strides(),
                                transpose_node->get_unroll_limit())
    {
    }

    // extends the tranpose nest
    jitted_transpose_node(
        std::shared_ptr<for_loop_node<ISA>> const&         for_node,
        std::shared_ptr<jitted_transpose_node<ISA>> const& transpose_jitter)
        : jitted_transpose_node(*transpose_jitter)
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
    }

    loop_tree_fn_type get_fn(std::map<std::string, int> const& tensors_idx,
                             std::map<std::string, int> const& sizes,
                             formulas_map_type const& formulas) const override
    {
        auto jit_fn = transposer_code_generator<std::conditional_t<
            std::is_same_v<ISA, avx512>, avx2_plus, ISA>>(
                          order, sizes, strides.at(output), strides.at(input),
                          64 /* unroll_limit */)
                          .get_shared();

        return
            [jit_fn, output_idx = tensors_idx.at(output),
             input_idx = tensors_idx.at(input)](
                std::vector<float*> const& tensors, std::vector<int> const&) {
                jit_fn(tensors[output_idx], tensors[input_idx]);
            };
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

template <class ISA>
loop_tree_node_ptr<ISA>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>> const&         node,
                       std::shared_ptr<jitted_loop_nest_node<ISA>> const& child)
{
    return loop_tree_node_ptr<ISA>(new jitted_loop_nest_node<ISA>(node, child));
}

template <class ISA>
loop_tree_node_ptr<ISA>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>> const& node,
                       std::shared_ptr<compute_node<ISA>> const&  child)
{
    return loop_tree_node_ptr<ISA>(new jitted_loop_nest_node<ISA>(node, child));
}

template <class ISA>
loop_tree_node_ptr<ISA>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>> const&  node,
                       std::shared_ptr<transpose_node<ISA>> const& child)
{
    return loop_tree_node_ptr<ISA>(new jitted_transpose_node<ISA>(node, child));
}

template <class ISA>
loop_tree_node_ptr<ISA>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>> const&         node,
                       std::shared_ptr<jitted_transpose_node<ISA>> const& child)
{
    return loop_tree_node_ptr<ISA>(new jitted_transpose_node<ISA>(node, child));
}

template <class ISA>
loop_tree_node_ptr<ISA> simplify_loop_nests(loop_tree_node_ptr<ISA> const& node,
                                            int current_depth         = 0,
                                            int max_interpreted_depth = 0)
{
    if (node->get_type() != node_kind::for_loop)
    {
        return node;
    }

    std::vector<loop_tree_node_ptr<ISA>> new_children;
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

    auto for_node = std::dynamic_pointer_cast<for_loop_node<ISA>>(node);
    loop_tree_node_ptr<ISA> single_child = new_children.at(0);

    switch (single_child->get_type())
    {
    case node_kind::for_loop:
        // child is not jitted, so can't add on this node
        return node;
        break;

    case node_kind::compute:
        return merge_loop_into_jitter(
            for_node,
            std::dynamic_pointer_cast<compute_node<ISA>>(single_child));
        break;

    case node_kind::jitted_loop_nest:
        return merge_loop_into_jitter(
            for_node, std::dynamic_pointer_cast<jitted_loop_nest_node<ISA>>(
                          single_child));
        break;

    case node_kind::transpose:
        return merge_loop_into_jitter(
            for_node,
            std::dynamic_pointer_cast<transpose_node<ISA>>(single_child));
        break;

    case node_kind::jitted_transpose:
        return merge_loop_into_jitter(
            for_node, std::dynamic_pointer_cast<jitted_transpose_node<ISA>>(
                          single_child));
        break;

    default:
        throw std::runtime_error("Unhandled node kind");
    }
}

template <class ISA>
inline std::string dump_recursively(loop_tree_node_ptr<ISA> const&    node,
                                    formulas_map_type const&          formulas,
                                    std::map<std::string, int> const& sizes,
                                    std::string&                      indent)
{
    std::ostringstream ss;
    ss << node->dump(formulas, sizes, indent);
    ss << std::endl;
    if (node->get_type() == node_kind::for_loop)
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

inline std::int64_t get_tensor_size(std::string const&                name,
                                    strides_map_type const&           strides,
                                    std::map<std::string, int> const& sizes,
                                    formulas_map_type const&          formulas)
{
    std::int64_t size = 1;
    for (auto const& s : sizes)
    {
        if (formulas.at(name).count(s.first))
            size += (s.second - 1) * strides.at(name).at(s.first);
    }
    size *= 4;
    return size;
}

template <class ISA>
std::int64_t get_largest_intermediate_output_size(
    loop_tree_node_ptr<ISA> const&    node,
    std::vector<std::string> const&   provided_tensors,
    std::map<std::string, int> const& sizes, formulas_map_type const& formulas)
{
    std::int64_t max_size = 0;
    switch (node->get_type())
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
    case node_kind::jitted_loop_nest:
        // fall through
    case node_kind::jitted_transpose:
        std::set<std::string> possible_intermediates =
            node->get_output_tensors();

        for (auto const& name : provided_tensors)
        {
            possible_intermediates.erase(name);
        }

        for (auto const& name : possible_intermediates)
        {
            max_size = std::max(
                max_size, get_tensor_size(name, node->get_tensor_strides(),
                                          sizes, formulas));
        }

        return max_size;
        break;
        // default:
        // throw std::runtime_error("Unhandled node kind");
    }
}

template <class ISA>
class loop_tree_program
{
private:
    std::vector<loop_tree_node_ptr<ISA>> nodes;
    std::map<std::string, int>           sizes;
    formulas_map_type                    formulas;
    // for forcing partially interpreted trees (mainly for testing)
    int max_interpreted_depth;

    // map tensor names to indices
    std::map<std::string, int> tensors_idx;

public:
    loop_tree_program(std::vector<loop_tree_node_ptr<ISA>> const& nodes,
                      std::map<std::string, int> const&           sizes,
                      formulas_map_type const&                    formulas,
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
        std::vector<loop_tree_node_ptr<ISA>> new_nodes;
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

    std::vector<loop_tree_node_ptr<ISA>> const& get_children() { return nodes; }

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

    std::function<void(std::map<std::string, float*>)> get_fn() const
    {
        std::vector<loop_tree_fn_type> sub_functions;
        // added to alpha at runtime to handle tensor initialization
        std::vector<int> alpha_offsets(tensors_idx.size());

        for (auto const& c : this->nodes)
        {
            sub_functions.push_back(c->get_fn(tensors_idx, sizes, formulas));
            for (auto const& t : c->get_output_tensors())
            {
                alpha_offsets[tensors_idx.at(t)] = 0;
            }
        }

        return [sub_functions, alpha_offsets, tensors_idx = this->tensors_idx](
                   std::map<std::string, float*> const& tensors) {
            std::vector<float*> tensors_vec(tensors_idx.size());
            for (auto const& e : tensors)
            {
                int idx          = tensors_idx.at(e.first);
                tensors_vec[idx] = e.second;
            }

            for (auto const& f : sub_functions)
            {
                f(tensors_vec, alpha_offsets);
            }
        };
    }
};

template <class ISA>
std::shared_ptr<loop_tree_program<ISA>>
make_loop_tree_program(std::vector<loop_tree_node_ptr<ISA>> const& nodes,
                       std::map<std::string, int> const&           sizes,
                       formulas_map_type const&                    formulas,
                       std::optional<int> max_interpreted_depth = std::nullopt)
{
    return std::shared_ptr<loop_tree_program<ISA>>(new loop_tree_program<ISA>(
        nodes, sizes, formulas, max_interpreted_depth));
}

} // namespace loop_tree
} // namespace dabun
