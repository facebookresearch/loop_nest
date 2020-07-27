// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "log.h"
#include "loop_nest.h"
#include "transposer.h"

namespace facebook
{
namespace sysml
{
namespace aot
{

enum class node_kind
{
    for_loop_node_type,
    compute_node_type,
    transpose_node_type,
    jitted_loop_nest_node_type,
    jitted_transpose_node_type
};

// Note: add classes from arithmetic_operations.h
// as needed
enum class arithmetic_op_kind
{
    plus,
    multiplies,
    max,
    min
};

std::shared_ptr<operation_pair_base>
get_operation_pair(arithmetic_op_kind plus_op, arithmetic_op_kind multiplies_op)
{

    std::map<std::pair<arithmetic_op_kind, arithmetic_op_kind>,
             std::shared_ptr<operation_pair_base>>
        op_map = {
            {{arithmetic_op_kind::plus, arithmetic_op_kind::multiplies},
             std::make_shared<operation_pair<basic_plus, basic_multiplies>>()},
            {{arithmetic_op_kind::max, arithmetic_op_kind::multiplies},
             std::make_shared<operation_pair<max, basic_multiplies>>()},
            {{arithmetic_op_kind::min, arithmetic_op_kind::multiplies},
             std::make_shared<operation_pair<min, basic_multiplies>>()},
            {{arithmetic_op_kind::max, arithmetic_op_kind::plus},
             std::make_shared<operation_pair<max, basic_plus>>()}};

    return op_map.at({plus_op, multiplies_op});
}

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

// void (map from name to tensors, map from name to alpha for given tensor)
using loop_tree_fn_type = std::function<void(
    std::map<std::string, float*> const&, std::map<std::string, int> const&)>;

template <class ISA>
class loop_tree_node
{
private:
    node_kind                                         kind_;
    std::vector<std::shared_ptr<loop_tree_node<ISA>>> children_;

public:
    virtual ~loop_tree_node(){};

    explicit loop_tree_node(node_kind kind)
        : kind_(kind)
    {
    }

    std::vector<std::shared_ptr<loop_tree_node<ISA>>> const&
    get_children() const
    {
        return children_;
    }

    void set_children(
        std::vector<std::shared_ptr<loop_tree_node<ISA>>> const& children)
    {
        children_ = children;
    }

    void
    set_children(std::vector<std::shared_ptr<loop_tree_node<ISA>>>&& children)
    {
        children_ = std::move(children);
    }

    node_kind get_type() const { return kind_; }

    // sizes, and tensor formulas
    virtual loop_tree_fn_type
    get_fn(std::map<std::string, int> const&,
           std::map<std::string, std::set<std::string>> const&) const = 0;

    virtual std::vector<std::string> get_tensors_used() const = 0;

    virtual std::vector<std::string> get_output_tensors() const = 0;

    virtual std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const = 0;
};

template <class ISA>
class compute_node : public loop_tree_node<ISA>
{
private:
    using super_type = loop_tree_node<ISA>;

    // 0 -> A, 1 -> B, rest are followed tensors
    std::vector<std::string>                          inputs;
    std::string                                       output;
    std::map<std::string, std::map<std::string, int>> strides;
    arithmetic_op_kind                                plus;
    arithmetic_op_kind                                multiplies;
    std::optional<int>                                unroll_limit;
    // TODO(j): need to add elementwise ops here...
    std::shared_ptr<elementwise_operation<ISA>> elementwise_preop;
    std::vector<std::string>                    elementwise_preop_tensors;
    std::shared_ptr<elementwise_operation<ISA>> elementwise_postop;
    std::vector<std::string>                    elementwise_postop_tensors;

public:
    compute_node(
        std::vector<std::string> const& inputs, std::string const& output,
        std::map<std::string, std::map<std::string, int>> const& strides,
        arithmetic_op_kind plus, arithmetic_op_kind multiplies,
        std::optional<int>                          unroll_limit = std::nullopt,
        std::shared_ptr<elementwise_operation<ISA>> elementwise_preop = nullptr,
        std::vector<std::string> elementwise_preop_tensors            = {},
        std::shared_ptr<elementwise_operation<ISA>> elementwise_postop =
            nullptr,
        std::vector<std::string> elementwise_postop_tensors = {})
        : super_type(node_kind::compute_node_type)
        , inputs(inputs)
        , output(output)
        , strides(strides)
        , plus(plus)
        , multiplies(multiplies)
        , unroll_limit(unroll_limit)
        , elementwise_preop(elementwise_preop)
        , elementwise_preop_tensors(elementwise_preop_tensors)
        , elementwise_postop(elementwise_postop)
        , elementwise_postop_tensors(elementwise_postop_tensors)
    {
        for (auto const& t : inputs)
        {
            assert(strides.count(t) > 0);
        }

        assert(strides.count(output));

        if (!elementwise_preop_tensors.empty())
        {
            assert(elementwise_preop != nullptr);
        }

        for (auto const& t : elementwise_preop_tensors)
        {
            assert(strides.count(t) > 0);
        }

        if (!elementwise_postop_tensors.empty())
        {
            assert(elementwise_postop != nullptr);
        }

        for (auto const& t : elementwise_postop_tensors)
        {
            assert(strides.count(t) > 0);
        }
    }

    std::string const& get_output() const { return output; }

    std::vector<std::string> const& get_inputs() const { return inputs; }

    arithmetic_op_kind get_plus() const { return plus; }

    arithmetic_op_kind get_multiplies() const { return multiplies; }

    std::optional<int> get_unroll_limit() const { return unroll_limit; }

    std::shared_ptr<elementwise_operation<ISA>> get_elementwise_preop()
    {
        return elementwise_preop;
    }

    std::vector<std::string> const& get_elementwise_preop_tensors()
    {
        return elementwise_preop_tensors;
    }

    std::shared_ptr<elementwise_operation<ISA>> get_elementwise_postop()
    {
        return elementwise_postop;
    }

    std::vector<std::string> const& get_elementwise_postop_tensors()
    {
        return elementwise_postop_tensors;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    std::vector<std::string> get_tensors_used() const
    {
        auto tensors_used = inputs;
        tensors_used.push_back(output);

        tensors_used.insert(tensors_used.end(),
                            elementwise_preop_tensors.begin(),
                            elementwise_preop_tensors.end());
        tensors_used.insert(tensors_used.end(),
                            elementwise_postop_tensors.begin(),
                            elementwise_postop_tensors.end());

        return tensors_used;
    }

    std::vector<std::string> get_output_tensors() const { return {output}; }

    loop_tree_fn_type
    get_fn(std::map<std::string, int> const& sizes,
           std::map<std::string, std::set<std::string>> const&) const
    {
        // TODO(j): if we want to support more ops, extend here otherwise only
        // supported through loop nest
        assert("Interpreted compute only supports FMA" &&
               (plus == arithmetic_op_kind::plus &&
                multiplies == arithmetic_op_kind::multiplies));

        assert("Interpreted doesn't support pre/post-ops" &&
               (elementwise_preop_tensors.empty() &&
                elementwise_postop_tensors.empty()));

        return [inputs = this->inputs, output = this->output](
                   std::map<std::string, float*> const& tensors,
                   std::map<std::string, int> const&    alphas) {
            assert(tensors.count(inputs[0]) && tensors.count(inputs[1]) &&
                   tensors.count(output));

            float* A = tensors.at(inputs[0]);
            float* B = tensors.at(inputs[1]);
            float* C = tensors.at(output);

            if (alphas.at(output) == 0)
            {
                C[0] = 0.0;
            }

            C[0] += A[0] * B[0];
        };
    }
};

template <class ISA>
class transpose_node : public loop_tree_node<ISA>
{

private:
    using super_type = loop_tree_node<ISA>;

    std::string                                       input;
    std::string                                       output;
    std::map<std::string, std::map<std::string, int>> strides;
    std::optional<int>                                unroll_limit;

public:
    transpose_node(std::string input, std::string output,
                   std::map<std::string, std::map<std::string, int>> strides,
                   std::optional<int> unroll_limit = std::nullopt)
        : super_type(node_kind::transpose_node_type)
        , input(input)
        , output(output)
        , strides(strides)
        , unroll_limit(unroll_limit)
    {
    }

    std::string const& get_input() const { return input; }

    std::string const& get_output() const { return output; }

    std::optional<int> get_unroll_limit() const { return unroll_limit; }

    std::vector<std::string> get_tensors_used() const
    {
        return {input, output};
    }

    std::vector<std::string> get_output_tensors() const { return {output}; }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    loop_tree_fn_type
    get_fn(std::map<std::string, int> const&,
           std::map<std::string, std::set<std::string>> const&) const
    {
        return [input  = this->input,
                output = this->output](std::map<std::string, float*> tensors,
                                       std::map<std::string, int> const&) {
            float* A = tensors.at(input);
            float* C = tensors.at(output);

            C[0] = A[0];
        };
    }
};

template <class ISA>
class for_loop_node : public loop_tree_node<ISA>
{
private:
    using super_type = loop_tree_node<ISA>;

    std::string var;
    int         delta;

    std::vector<std::string> in_scope_tensor_names;
    std::vector<std::string> in_scope_output_tensor_names;
    std::map<std::string, std::map<std::string, int>> in_scope_tensor_strides;

private:
    void set_in_scope_tensor_info()
    {
        for (auto c : this->get_children())
        {
            auto node_tensor_names = c->get_tensors_used();
            in_scope_tensor_names.insert(in_scope_tensor_names.end(),
                                         node_tensor_names.begin(),
                                         node_tensor_names.end());

            auto node_output_tensor_names = c->get_output_tensors();
            in_scope_output_tensor_names.insert(
                in_scope_output_tensor_names.end(),
                node_output_tensor_names.begin(),
                node_output_tensor_names.end());

            auto node_tensor_strides = c->get_tensor_strides();

            in_scope_tensor_strides.insert(node_tensor_strides.begin(),
                                           node_tensor_strides.end());
        }
    }

    std::function<void(std::map<std::string, float*>&)>
    get_tensor_advancer(std::vector<std::string> const& tensor_names) const
    {
        std::vector<std::pair<std::string, std::int64_t>> to_advance;

        for (auto const& name : tensor_names)
        {
            if (in_scope_tensor_strides.at(name).count(var))
            {
                std::int64_t offset =
                    in_scope_tensor_strides.at(name).at(var) * delta;
                to_advance.push_back({name, offset});
            }
        }

        return [=](std::map<std::string, float*>& tensors) {
            for (auto const& p : to_advance)
            {
                assert(tensors.count(p.first));
                tensors[p.first] += p.second;
            }
        };
    }

    std::function<void(std::map<std::string, int>&, int)> get_alphas_adjuster(
        std::vector<std::string> const&                     output_tensor_names,
        std::map<std::string, std::set<std::string>> const& formulas) const
    {

        std::vector<std::string> to_adjust;
        for (auto const& name : output_tensor_names)
        {
            if (formulas.count(name) && formulas.at(name).count(var) == 0)
            {
                // reduction variable, so adjust the tensor's alpha
                to_adjust.push_back(name);
            }
        }

        return [=](std::map<std::string, int>& alphas, int adjustment) {
            for (auto const& name : to_adjust)
            {
                assert(alphas.count(name));
                alphas[name] += adjustment;
            }
        };
    }

public:
    std::string const& get_var() const { return var; }
    int                get_delta() const { return delta; }

    for_loop_node(std::string var, int delta,
                  std::vector<std::shared_ptr<loop_tree_node<ISA>>> children)
        : super_type(node_kind::for_loop_node_type)
        , var(var)
        , delta(delta)
    {

        this->set_children(children);
        set_in_scope_tensor_info();
    }

    std::vector<std::string> get_tensors_used() const override
    {
        return in_scope_tensor_names;
    }

    std::vector<std::string> get_output_tensors() const
    {
        return in_scope_output_tensor_names;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const override
    {
        return in_scope_tensor_strides;
    }

    loop_tree_fn_type
    get_fn(std::map<std::string, int> const&                   sizes,
           std::map<std::string, std::set<std::string>> const& formulas) const
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
                full_fns.push_back(c->get_fn(s, formulas));
            }
            if (rest)
            {
                auto s = sizes;
                s[var] = rest;
                tail_fns.push_back(c->get_fn(s, formulas));
            }
        }

        auto advancer = get_tensor_advancer(get_tensors_used());
        auto alpha_adjuster =
            get_alphas_adjuster(get_output_tensors(), formulas);

        return [=](std::map<std::string, float*> tensors,
                   std::map<std::string, int>    alphas) {
            for (int i = 0; i < full; ++i)
            {
                for (auto const& fn : full_fns)
                {
                    fn(tensors, alphas);
                }
                advancer(tensors);
                alpha_adjuster(alphas, 2);
            }

            for (auto const& fn : tail_fns)
            {
                fn(tensors, alphas);
            }
        };
    }
};

template <class ISA>
class jitted_loop_nest_node : public loop_tree_node<ISA>
{
private:
    using super_type = loop_tree_node<ISA>;

    std::vector<std::string>                          inputs;
    std::string                                       output;
    std::vector<std::pair<std::string, int>>          order;
    std::map<std::string, std::map<std::string, int>> strides;
    arithmetic_op_kind                                plus;
    arithmetic_op_kind                                multiplies;
    std::optional<int>                                unroll_limit;
    std::shared_ptr<elementwise_operation<ISA>>       elementwise_preop;
    std::vector<std::string>                          elementwise_preop_tensors;
    std::shared_ptr<elementwise_operation<ISA>>       elementwise_postop;
    std::vector<std::string> elementwise_postop_tensors;

public:
    jitted_loop_nest_node(
        std::vector<std::string> inputs, std::string output,
        std::vector<std::pair<std::string, int>>          order,
        std::map<std::string, std::map<std::string, int>> strides,
        arithmetic_op_kind plus, arithmetic_op_kind multiplies,
        std::optional<int>                          unroll_limit = std::nullopt,
        std::shared_ptr<elementwise_operation<ISA>> elementwise_preop = nullptr,
        std::vector<std::string> elementwise_preop_tensors            = {},
        std::shared_ptr<elementwise_operation<ISA>> elementwise_postop =
            nullptr,
        std::vector<std::string> elementwise_postop_tensors = {})
        : super_type(node_kind::jitted_loop_nest_node_type)
        , inputs(inputs)
        , output(output)
        , order(order)
        , strides(strides)
        , plus(plus)
        , multiplies(multiplies)
        , unroll_limit(unroll_limit)
        , elementwise_preop(elementwise_preop)
        , elementwise_preop_tensors(elementwise_preop_tensors)
        , elementwise_postop(elementwise_postop)
        , elementwise_postop_tensors(elementwise_postop_tensors)
    {
    }

    jitted_loop_nest_node(
        std::shared_ptr<for_loop_node<ISA>>         for_node,
        std::shared_ptr<jitted_loop_nest_node<ISA>> compute_jitter_node)
        : super_type(node_kind::jitted_loop_nest_node_type)
        , inputs(compute_jitter_node->inputs)
        , output(compute_jitter_node->output)
        , order(compute_jitter_node->order)
        , strides(compute_jitter_node->strides)
        , plus(compute_jitter_node->plus)
        , multiplies(compute_jitter_node->multiplies)
        , unroll_limit(compute_jitter_node->unroll_limit)
        , elementwise_preop(compute_jitter_node->elementwise_preop)
        , elementwise_preop_tensors(
              compute_jitter_node->elementwise_preop_tensors)
        , elementwise_postop(compute_jitter_node->elementwise_postop)
        , elementwise_postop_tensors(
              compute_jitter_node->elementwise_postop_tensors)
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
    }

    jitted_loop_nest_node(std::shared_ptr<for_loop_node<ISA>> for_node,
                          std::shared_ptr<compute_node<ISA>>  compute_node)
        : super_type(node_kind::jitted_loop_nest_node_type)
        , inputs(compute_node->get_inputs())
        , output(compute_node->get_output())
        , order({{for_node->get_var(), for_node->get_delta()}})
        , strides(compute_node->get_tensor_strides())
        , plus(compute_node->get_plus())
        , multiplies(compute_node->get_multiplies())
        , unroll_limit(compute_node->get_unroll_limit())
        , elementwise_preop(compute_node->get_elementwise_preop())
        , elementwise_preop_tensors(
              compute_node->get_elementwise_preop_tensors())
        , elementwise_postop(compute_node->get_elementwise_postop())
        , elementwise_postop_tensors(
              compute_node->get_elementwise_postop_tensors())
    {
    }

    loop_tree_fn_type
    get_fn(std::map<std::string, int> const&                   sizes,
           std::map<std::string, std::set<std::string>> const& formulas) const
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

        // loop nest currently supports at most 2 followed tensors for
        // elementwise ops
        assert(extra_tensors.size() <= 2);

        auto jit_fn =
            facebook::sysml::aot::FMA_loop_nest_jitter<ISA>(
                order, sizes, formulas.at(output), formulas.at(inputs[0]),
                formulas.at(inputs[1]), strides.at(output),
                strides.at(inputs[0]), strides.at(inputs[1]),
                get_operation_pair(plus, multiplies), unroll_limit,
                elementwise_preop, preop_strides, elementwise_postop,
                postop_strides)
                .get_shared();

        auto output = this->output;
        auto inputs = this->inputs;

        if (extra_tensors.size() == 0)
        {
            return [jit_fn, inputs,
                    output](std::map<std::string, float*>     tensors,
                            std::map<std::string, int> const& alphas) {
                jit_fn(tensors.at(output), tensors.at(inputs[0]),
                       tensors.at(inputs[1]), alphas.at(output));
            };
        }
        else if (extra_tensors.size() == 1)
        {
            auto jit_fn_cast =
                aot_fn_cast<void(float*, float const*, float const*, int,
                                 float const*)>(std::move(jit_fn));

            return [jit_fn_cast, inputs, output,
                    extra_tensors](std::map<std::string, float*>     tensors,
                                   std::map<std::string, int> const& alphas) {
                jit_fn_cast(tensors.at(output), tensors.at(inputs[0]),
                            tensors.at(inputs[1]), alphas.at(output),
                            tensors.at(extra_tensors[0]));
            };
        }
        else if (extra_tensors.size() == 2)
        {
            auto jit_fn_cast =
                aot_fn_cast<void(float*, float const*, float const*, int,
                                 float const*, float const*)>(
                    std::move(jit_fn));

            return [jit_fn_cast, inputs, output,
                    extra_tensors](std::map<std::string, float*>     tensors,
                                   std::map<std::string, int> const& alphas) {
                jit_fn_cast(tensors.at(output), tensors.at(inputs[0]),
                            tensors.at(inputs[1]), alphas.at(output),
                            tensors.at(extra_tensors[0]),
                            tensors.at(extra_tensors[1]));
            };
        }
        else
        {
            assert("Exceeded number of allowed followed tensors" && false);
        }
    }

    std::vector<std::string> get_tensors_used() const override
    {
        std::vector<std::string> tensors_used = inputs;
        tensors_used.push_back(output);
        return tensors_used;
    }

    std::vector<std::string> get_output_tensors() const { return {output}; }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const override
    {
        return strides;
    }
};

template <class ISA>
class jitted_transpose_node : public loop_tree_node<ISA>
{
private:
    using super_type = loop_tree_node<ISA>;

    std::string                                       input;
    std::string                                       output;
    std::vector<std::pair<std::string, int>>          order;
    std::map<std::string, std::map<std::string, int>> strides;
    std::optional<int>                                unroll_limit;

public:
    jitted_transpose_node(
        std::string input, std::string output,
        std::vector<std::pair<std::string, int>>          order,
        std::map<std::string, std::map<std::string, int>> strides,
        std::optional<int> unroll_limit = std::nullopt)
        : super_type(node_kind::jitted_transpose_node_type)
        , input(input)
        , output(output)
        , order(order)
        , strides(strides)
        , unroll_limit(unroll_limit)
    {
    }

    jitted_transpose_node(std::shared_ptr<for_loop_node<ISA>>  for_node,
                          std::shared_ptr<transpose_node<ISA>> transpose_node)
        : super_type(node_kind::jitted_transpose_node_type)
        , input(transpose_node->get_input())
        , output(transpose_node->get_output())
        , order({})
        , strides(transpose_node->get_tensor_strides())
        , unroll_limit(transpose_node->get_unroll_limit())
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
    }

    jitted_transpose_node(
        std::shared_ptr<for_loop_node<ISA>>         for_node,
        std::shared_ptr<jitted_transpose_node<ISA>> transpose_jitter)
        : super_type(node_kind::jitted_transpose_node_type)
        , input(transpose_jitter->input)
        , output(transpose_jitter->output)
        , order(transpose_jitter->order)
        , strides(transpose_jitter->strides)
        , unroll_limit(transpose_jitter->unroll_limit)
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
    }

    loop_tree_fn_type
    get_fn(std::map<std::string, int> const&                   sizes,
           std::map<std::string, std::set<std::string>> const& formulas) const
    {
        // TODO(j): needs user unroll limit
        auto jit_fn = facebook::sysml::aot::transposer_jitter<ISA>(
                          order, sizes, strides.at(output), strides.at(input))
                          .get_shared();

        auto output = this->output;
        auto input  = this->input;

        return
            [jit_fn, output, input](std::map<std::string, float*>     tensors,
                                    std::map<std::string, int> const& alphas) {
                jit_fn(tensors.at(output), tensors.at(input));
            };
    }

    std::vector<std::string> get_tensors_used() const override
    {
        return {input, output};
    }

    std::vector<std::string> get_output_tensors() const { return {output}; }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const override
    {
        return strides;
    }
};

template <class ISA>
std::shared_ptr<loop_tree_node<ISA>>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>>         node,
                       std::shared_ptr<jitted_loop_nest_node<ISA>> child)
{
    return std::shared_ptr<loop_tree_node<ISA>>(
        new jitted_loop_nest_node<ISA>(node, child));
}

template <class ISA>
std::shared_ptr<loop_tree_node<ISA>>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>> node,
                       std::shared_ptr<compute_node<ISA>>  child)
{
    return std::shared_ptr<loop_tree_node<ISA>>(
        new jitted_loop_nest_node<ISA>(node, child));
}

template <class ISA>
std::shared_ptr<loop_tree_node<ISA>>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>>  node,
                       std::shared_ptr<transpose_node<ISA>> child)
{
    return std::shared_ptr<loop_tree_node<ISA>>(
        new jitted_transpose_node<ISA>(node, child));
}

template <class ISA>
std::shared_ptr<loop_tree_node<ISA>>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>>         node,
                       std::shared_ptr<jitted_transpose_node<ISA>> child)
{
    return std::shared_ptr<loop_tree_node<ISA>>(
        new jitted_transpose_node<ISA>(node, child));
}

#ifdef TEST_STOP_SIMPLIFICATION
int test_simplification_counter_ = 0;
#endif

template <class ISA>
std::shared_ptr<loop_tree_node<ISA>>
simplify_loop_nests(std::shared_ptr<loop_tree_node<ISA>> node)
{
    if (node->get_type() != node_kind::for_loop_node_type)
    {
        return node;
    }

    std::vector<std::shared_ptr<loop_tree_node<ISA>>> new_children;
    for (auto c : node->get_children())
    {
        new_children.push_back(simplify_loop_nests(c));
    }
    node->set_children(new_children);

    // can't merge into loop nest compute or loop nest transpose
    // since has "split"
    if (new_children.size() > 1)
    {
        return node;
    }

#ifdef TEST_STOP_SIMPLIFICATION
    if (test_simplification_counter_ >= 3)
    {
        std::cout << "Stopping short on simplification" << std::endl;
        return node;
    }
#endif

    auto for_node = std::dynamic_pointer_cast<for_loop_node<ISA>>(node);
    std::shared_ptr<loop_tree_node<ISA>> single_child = new_children.at(0);

    switch (single_child->get_type())
    {
    case node_kind::compute_node_type:
#ifdef TEST_STOP_SIMPLIFICATION
        test_simplification_counter_ += 1;
#endif

        return merge_loop_into_jitter(
            for_node,
            std::dynamic_pointer_cast<compute_node<ISA>>(single_child));
        break;

    case node_kind::jitted_loop_nest_node_type:
        return merge_loop_into_jitter(
            for_node, std::dynamic_pointer_cast<jitted_loop_nest_node<ISA>>(
                          single_child));
        break;

    case node_kind::transpose_node_type:
        return merge_loop_into_jitter(
            for_node,
            std::dynamic_pointer_cast<transpose_node<ISA>>(single_child));
        break;

    case node_kind::jitted_transpose_node_type:
        return merge_loop_into_jitter(
            for_node, std::dynamic_pointer_cast<jitted_transpose_node<ISA>>(
                          single_child));
        break;

    default:
        assert("Unhandled node kind" && false);
        return node;
    }
}

std::int64_t get_tensor_size(
    std::string const&                                       name,
    std::map<std::string, std::map<std::string, int>> const& strides,
    std::map<std::string, int> const&                        sizes,
    std::map<std::string, std::set<std::string>> const&      formulas)
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
    std::shared_ptr<loop_tree_node<ISA>>                node,
    std::vector<std::string> const&                     provided_tensors,
    std::map<std::string, int> const&                   sizes,
    std::map<std::string, std::set<std::string>> const& formulas)
{
    std::int64_t max_size = 0;
    switch (node->get_type())
    {
    case node_kind::for_loop_node_type:
        for (auto const& child : node->get_children())
        {
            max_size =
                std::max(max_size, get_largest_intermediate_output_size(child));
        }
        return max_size;
        break;
    case node_kind::compute_node_type:
        // fall through
    case node_kind::transpose_node_type:
        // fall through
    case node_kind::jitted_loop_nest_node_type:
        // fall through
    case node_kind::jitted_transpose_node_type:
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
    default:
        assert("Unhandled node kind" && false);
    }
}

template <class ISA>
class loop_tree_program
{
private:
    std::vector<std::shared_ptr<loop_tree_node<ISA>>> nodes;
    std::map<std::string, int>                        sizes;
    std::map<std::string, std::set<std::string>>      formulas;
    std::set<std::string>                             provided_tensors;

    static std::vector<std::shared_ptr<loop_tree_node<ISA>>>
    loop_nest_compute_to_tree(
        std::vector<std::pair<std::string, int>> order,
        std::map<std::string, int>               C_strides,
        std::map<std::string, int>               A_strides,
        std::map<std::string, int> B_strides, std::optional<int> unroll_limit,
        std::shared_ptr<elementwise_operation<ISA>> elementwise_preop,
        std::vector<std::map<std::string, int>> const&
                                                    elementwise_preop_strides,
        std::shared_ptr<elementwise_operation<ISA>> elementwise_postop,
        std::vector<std::map<std::string, int>> const&
            elementwise_postop_strides)
    {

        std::map<std::string, std::map<std::string, int>> tensor_strides = {
            {"A", A_strides}, {"B", B_strides}, {"C", C_strides}};

        assert("Convenience wrapper handles single followed pre-op tensor" &&
               (elementwise_preop_strides.size() <= 1));
        assert("Convenience wrapper handles single followed post-op tensor" &&
               (elementwise_postop_strides.size() <= 1));

        std::vector<std::string> preop_tensors;
        std::vector<std::string> postop_tensors;

        if (elementwise_preop != nullptr &&
            (!elementwise_preop_strides.empty()))
        {
            tensor_strides["pre"] = elementwise_preop_strides[0];
            preop_tensors.push_back("pre");
        }

        if (elementwise_postop != nullptr &&
            (!elementwise_postop_strides.empty()))
        {
            tensor_strides["post"] = elementwise_postop_strides[0];
            postop_tensors.push_back("post");
        }

        auto innermost =
            std::shared_ptr<compute_node<ISA>>(new compute_node<ISA>(
                {"A", "B"}, "C", tensor_strides, arithmetic_op_kind::plus,
                arithmetic_op_kind::multiplies, unroll_limit, elementwise_preop,
                preop_tensors, elementwise_postop, postop_tensors));

        std::shared_ptr<loop_tree_node<ISA>> current = innermost;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            auto new_node = std::shared_ptr<for_loop_node<ISA>>(
                new for_loop_node<ISA>(it->first, it->second, {current}));
            current = new_node;
        }

        return {current};
    }

    std::vector<std::shared_ptr<loop_tree_node<ISA>>>
    loop_nest_transpose_to_tree(std::vector<std::pair<std::string, int>> order,
                                std::map<std::string, int> C_strides,
                                std::map<std::string, int> A_strides,
                                std::optional<int>         unroll_limit)
    {
        std::shared_ptr<transpose_node<ISA>> innermost =
            std::shared_ptr<transpose_node<ISA>>(new transpose_node<ISA>(
                "A", "C", {{"A", A_strides}, {"C", C_strides}}, unroll_limit));

        std::shared_ptr<loop_tree_node<ISA>> current = innermost;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            std::shared_ptr<loop_tree_node<ISA>> new_node =
                std::shared_ptr<for_loop_node<ISA>>(
                    new for_loop_node<ISA>(it->first, it->second, {current}));
            current = new_node;
        }

        return {current};
    }

public:
    loop_tree_program(std::vector<std::shared_ptr<loop_tree_node<ISA>>> nodes,
                      std::map<std::string, int>                        sizes,
                      std::map<std::string, std::set<std::string>> formulas,
                      std::set<std::string> provided_tensors)
        : nodes(nodes)
        , sizes(sizes)
        , formulas(formulas)
        , provided_tensors(provided_tensors)
    {

#ifndef NOPTIM
        LN_LOG(DEBUG) << "Pass: Simplifying loop nests\n";
        std::vector<std::shared_ptr<loop_tree_node<ISA>>> new_nodes;
        for (auto c : nodes)
        {
            new_nodes.push_back(simplify_loop_nests(c));
        }
        nodes = new_nodes;
#endif
    }

    std::vector<std::shared_ptr<loop_tree_node<ISA>>> get_children()
    {
        return nodes;
    }

    loop_tree_program(
        std::vector<std::pair<std::string, int>> order,
        std::map<std::string, int> sizes, std::set<std::string> C_formula,
        std::set<std::string> A_formula, std::set<std::string> B_formula,
        std::map<std::string, int>                  C_strides,
        std::map<std::string, int>                  A_strides,
        std::map<std::string, int>                  B_strides,
        std::optional<int>                          unroll_limit = std::nullopt,
        std::shared_ptr<elementwise_operation<ISA>> elementwise_preop = nullptr,
        std::vector<std::map<std::string, int>> const&
                                                    elementwise_preop_strides = {},
        std::shared_ptr<elementwise_operation<ISA>> elementwise_postop =
            nullptr,
        std::vector<std::map<std::string, int>> const&
            elementwise_postop_strides = {})
        : loop_tree_program(
              loop_nest_compute_to_tree(
                  order, C_strides, A_strides, B_strides, unroll_limit,
                  elementwise_preop, elementwise_preop_strides,
                  elementwise_postop, elementwise_postop_strides),
              sizes, {{"C", C_formula}, {"A", A_formula}, {"B", B_formula}},
              {"A", "B", "C"})
    {
        if (!elementwise_preop_strides.empty())
        {
            provided_tensors.insert("pre");
        }
        if (!elementwise_postop_strides.empty())
        {
            provided_tensors.insert("post");
        }
    }

    loop_tree_program(std::vector<std::pair<std::string, int>> order,
                      std::map<std::string, int>               sizes,
                      std::map<std::string, int>               Out_strides,
                      std::map<std::string, int>               In_strides,
                      std::optional<int> unroll_limit = std::nullopt)
        : loop_tree_program(loop_nest_transpose_to_tree(
                                order, Out_strides, In_strides, unroll_limit),
                            sizes, {}, {"A", "C"})
    {
    }

    std::int64_t get_scratch_size() const
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

    loop_tree_fn_type get_fn() const
    {
        std::vector<loop_tree_fn_type> sub_functions;

        for (auto const& c : this->nodes)
        {
            sub_functions.push_back(c->get_fn(sizes, formulas));
        }

        return [sub_functions](std::map<std::string, float*> const& tensors,
                               std::map<std::string, int> const&    alphas) {
            for (auto const& f : sub_functions)
            {
                f(tensors, alphas);
            }
        };
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
