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

    virtual loop_tree_fn_type
    get_fn(std::map<std::string, int> const&) const = 0;

    virtual std::vector<std::string> get_tensors_used() const = 0;

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
    // TODO(j): need to add elementwise ops here...
    int AlphaReg_ = 0;

public:
    compute_node(
        std::vector<std::string> const& inputs, std::string const& output,
        std::map<std::string, std::map<std::string, int>> const& strides,
        arithmetic_op_kind plus, arithmetic_op_kind multiplies)
        : super_type(node_kind::compute_node_type)
        , inputs(inputs)
        , output(output)
        , strides(strides)
        , plus(plus)
        , multiplies(multiplies)
    {
    }

    std::string const&              get_output() const { return output; }
    std::vector<std::string> const& get_inputs() const { return inputs; }
    arithmetic_op_kind              get_plus() const { return plus; }
    arithmetic_op_kind get_multiplies() const { return multiplies; }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    std::vector<std::string> get_tensors_used() const
    {
        auto tensors_used = inputs;
        tensors_used.push_back(output);
        return tensors_used;
    }

    loop_tree_fn_type get_fn(std::map<std::string, int> const& sizes) const
    {
        // TODO(j): if we want to support more ops, extend here otherwise only
        // supported through loop nest
        assert("Interpreted compute only supports FMA" &&
               (plus == arithmetic_op_kind::plus &&
                multiplies == arithmetic_op_kind::multiplies));

        return [inputs = this->inputs, output = this->output](
                   std::map<std::string, float*> const& tensors,
                   std::map<std::string, int> const&    alphas) {
            LN_LOG(DEBUG) << "Hit compute\n";
            assert(tensors.count(inputs[0]) && tensors.count(inputs[1]) &&
                   tensors.count(output));

            float* A = tensors.at(inputs[0]);
            float* B = tensors.at(inputs[1]);
            float* C = tensors.at(output);

            if (alphas.at(output) == 0)
            {
                C[0] = 0.0;
            }

            LN_LOG(DEBUG) << "(A:" << A[0] << ") * (B:" << B[0] << ")"
                          << "\n";
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

public:
    transpose_node(std::string input, std::string output,
                   std::map<std::string, std::map<std::string, int>> strides)
        : super_type(node_kind::transpose_node_type)
        , input(input)
        , output(output)
        , strides(strides)
    {
    }

    std::string const& get_input() const { return input; }

    std::string const& get_output() const { return output; }

    std::vector<std::string> get_tensors_used() const
    {
        return {input, output};
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    loop_tree_fn_type get_fn(std::map<std::string, int> const&) const
    {
        return [input  = this->input,
                output = this->output](std::map<std::string, float*> tensors,
                                       std::map<std::string, int> const&) {
            LN_LOG(DEBUG) << "Hit transpose\n";
            float* A = tensors.at(input);
            float* C = tensors.at(output);
            LN_LOG(DEBUG) << "(C:" << C[0] << ") * (A:" << A[0] << ")"
                          << "\n";
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

            // TODO(j): collect output tensor names
            // in_scope_output_tensor_names

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
                LN_LOG(DEBUG) << name << "+=" << offset << "\n";
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

    std::vector<std::string> const& get_output_tensors() const
    {
        return in_scope_output_tensor_names;
    }

    std::function<void(std::map<std::string, int>&, int)> get_alphas_adjuster(
        std::vector<std::string> const& output_tensor_names) const
    {
        std::vector<std::string> to_adjust;
        for (auto const& name : output_tensor_names)
        {
            // TODO(j): collect tensor formulas
            // if (in_scope_tensor_formulas.at(name).count(var) == 0)
            // {
            //     // reduction variable, so adjust the tensor's alpha
            //     to_adjust.push_back(name);
            // }
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

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const override
    {
        return in_scope_tensor_strides;
    }

    loop_tree_fn_type get_fn(std::map<std::string, int> const& sizes) const
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
                full_fns.push_back(c->get_fn(s));
            }
            if (rest)
            {
                std::cout << "Issued a tail: " << rest << std::endl;
                auto s = sizes;
                s[var] = rest;
                tail_fns.push_back(c->get_fn(s));
            }
        }

        auto advancer       = get_tensor_advancer(get_tensors_used());
        auto alpha_adjuster = get_alphas_adjuster(get_output_tensors());

        return [=](std::map<std::string, float*>     tensors,
                   std::map<std::string, int>        alphas) {
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
            // alpha_adjuster(alphas, -2 * full);
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
    std::map<std::string, int>                        sizes;
    std::map<std::string, std::set<std::string>>      formulas;
    std::map<std::string, std::map<std::string, int>> strides;
    arithmetic_op_kind                                plus;
    arithmetic_op_kind                                multiplies;

public:
    jitted_loop_nest_node(
        std::vector<std::string> inputs, std::string output,
        std::vector<std::pair<std::string, int>>          order,
        std::map<std::string, int>                        sizes,
        std::map<std::string, std::set<std::string>>      formulas,
        std::map<std::string, std::map<std::string, int>> strides,
        arithmetic_op_kind plus, arithmetic_op_kind multiplies)
        : super_type(node_kind::jitted_loop_nest_node_type)
        , inputs(inputs)
        , output(output)
        , order(order)
        , sizes(sizes)
        , formulas(formulas)
        , strides(strides)
        , plus(plus)
        , multiplies(multiplies)
    {
    }

    jitted_loop_nest_node(
        std::shared_ptr<for_loop_node<ISA>>         for_node,
        std::shared_ptr<jitted_loop_nest_node<ISA>> compute_jitter_node)
        : super_type(node_kind::jitted_loop_nest_node_type)
        , inputs(compute_jitter_node->inputs)
        , output(compute_jitter_node->output)
        , order(compute_jitter_node->order)
        , sizes(compute_jitter_node->sizes)
        , formulas(compute_jitter_node->formulas)
        , strides(compute_jitter_node->strides)
        , plus(compute_jitter_node->plus)
        , multiplies(compute_jitter_node->multiplies)
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
    }

    jitted_loop_nest_node(std::shared_ptr<for_loop_node<ISA>> for_node,
                          std::shared_ptr<compute_node<ISA>>  compute_node,
                          std::map<std::string, int>          sizes,
                          std::map<std::string, std::set<std::string>> formulas)
        : super_type(node_kind::jitted_loop_nest_node_type)
        , inputs(compute_node->get_inputs())
        , output(compute_node->get_output())
        , order({{for_node->get_var(), for_node->get_delta()}})
        , sizes(sizes)
        , formulas(formulas)
        , strides(compute_node->get_tensor_strides())
        , plus(compute_node->get_plus())
        , multiplies(compute_node->get_multiplies())
    {
    }

    loop_tree_fn_type get_fn(std::map<std::string, int> const&) const
    {
        // TODO(j): call to jitter should reflect all other arguments (e.g. type
        // of op-pair etc)
        auto jit_fn =
            facebook::sysml::aot::FMA_loop_nest_jitter<ISA>(
                order, sizes, formulas.at(output), formulas.at(inputs[0]),
                formulas.at(inputs[1]), strides.at(output),
                strides.at(inputs[0]), strides.at(inputs[1]),
                get_operation_pair(plus, multiplies))
                .get_shared();

        auto output = this->output;
        auto inputs = this->inputs;

        return
            [jit_fn, inputs, output](std::map<std::string, float*>     tensors,
                                     std::map<std::string, int> const& alphas) {
                jit_fn(tensors.at(output), tensors.at(inputs[0]),
                       tensors.at(inputs[1]), alphas.at(output));
            };
    }

    std::vector<std::string> get_tensors_used() const override
    {
        std::vector<std::string> tensors_used = inputs;
        tensors_used.push_back(output);
        return tensors_used;
    }

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

public:
    jitted_transpose_node(
        std::string input, std::string output,
        std::vector<std::pair<std::string, int>>          order,
        std::map<std::string, std::map<std::string, int>> strides)
        : super_type(node_kind::jitted_transpose_node_type)
        , input(input)
        , output(output)
        , order(order)
        , strides(strides)
    {
    }

    jitted_transpose_node(std::shared_ptr<for_loop_node<ISA>>  for_node,
                          std::shared_ptr<transpose_node<ISA>> transpose_node)
        : super_type(node_kind::jitted_transpose_node_type)
        , input(transpose_node->get_input())
        , output(transpose_node->get_output())
        , order({})
        , strides(transpose_node->get_tensor_strides())
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
    {
        order.insert(order.begin(),
                     {for_node->get_var(), for_node->get_delta()});
    }

    loop_tree_fn_type get_fn(std::map<std::string, int> const& sizes) const
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
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>>          node,
                       std::shared_ptr<compute_node<ISA>>           child,
                       std::map<std::string, int>                   sizes,
                       std::map<std::string, std::set<std::string>> formulas)
{
    return std::shared_ptr<loop_tree_node<ISA>>(
        new jitted_loop_nest_node<ISA>(node, child, sizes, formulas));
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

template <class ISA>
std::shared_ptr<loop_tree_node<ISA>>
simplify_loop_nests(std::shared_ptr<loop_tree_node<ISA>>         node,
                    std::map<std::string, int>                   sizes,
                    std::map<std::string, std::set<std::string>> formulas)
{
    if (node->get_type() != node_kind::for_loop_node_type)
    {
        return node;
    }

    std::vector<std::shared_ptr<loop_tree_node<ISA>>> new_children;
    for (auto c : node->get_children())
    {
        new_children.push_back(simplify_loop_nests(c, sizes, formulas));
    }
    node->set_children(new_children);

    // can't merge into loop nest compute or loop nest transpose
    // since has "split"
    if (new_children.size() > 1)
    {
        return node;
    }

    auto for_node = std::dynamic_pointer_cast<for_loop_node<ISA>>(node);
    std::shared_ptr<loop_tree_node<ISA>> single_child = new_children.at(0);

    switch (single_child->get_type())
    {
    case node_kind::compute_node_type:
        return merge_loop_into_jitter(
            for_node,
            std::dynamic_pointer_cast<compute_node<ISA>>(single_child), sizes,
            formulas);
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
        assert("Unhandled merger" && false);
        return node;
    }
}

template <class ISA>
class loop_tree_program
{
private:
    std::vector<std::shared_ptr<loop_tree_node<ISA>>> nodes;
    std::map<std::string, int>                        sizes;
    std::map<std::string, std::set<std::string>>      formulas;

    static std::vector<std::shared_ptr<loop_tree_node<ISA>>>
    loop_nest_compute_to_tree(std::vector<std::pair<std::string, int>> order,
                              std::map<std::string, int> C_strides,
                              std::map<std::string, int> A_strides,
                              std::map<std::string, int> B_strides)
    {
        auto innermost =
            std::shared_ptr<compute_node<ISA>>(new compute_node<ISA>(
                {"A", "B"}, "C",
                {{"A", A_strides}, {"B", B_strides}, {"C", C_strides}},
                arithmetic_op_kind::plus, arithmetic_op_kind::multiplies));

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
                                std::map<std::string, int> A_strides)
    {
        std::shared_ptr<transpose_node<ISA>> innermost =
            std::shared_ptr<transpose_node<ISA>>(new transpose_node<ISA>(
                "A", "C", {{"A", A_strides}, {"C", C_strides}}));

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
    loop_tree_program(
        std::vector<std::shared_ptr<loop_tree_node<ISA>>> nodes,
        std::map<std::string, int>                        sizes,
        std::map<std::string, std::set<std::string>>      formulas = {})
        : nodes(nodes)
        , sizes(sizes)
        , formulas(formulas)
    {

        // LN_LOG(DEBUG) << "Pass: Simplifying loop nests\n";
        // std::vector<std::shared_ptr<loop_tree_node<ISA>>> new_nodes;
        // for (auto c : nodes)
        // {
        //     new_nodes.push_back(simplify_loop_nests(c, sizes, formulas));
        // }
        // nodes = new_nodes;
    }

    std::vector<std::shared_ptr<loop_tree_node<ISA>>> get_children()
    {
        return nodes;
    }

    loop_tree_program(std::vector<std::pair<std::string, int>> order,
                      std::map<std::string, int>               sizes,
                      std::set<std::string>                    C_formula,
                      std::set<std::string>                    A_formula,
                      std::set<std::string>                    B_formula,
                      std::map<std::string, int>               C_strides,
                      std::map<std::string, int>               A_strides,
                      std::map<std::string, int>               B_strides)
        : loop_tree_program(
              loop_nest_compute_to_tree(order, C_strides, A_strides, B_strides),
              sizes, {{"C", C_formula}, {"A", A_formula}, {"B", B_formula}})
    {
    }

    loop_tree_program(std::vector<std::pair<std::string, int>> order,
                      std::map<std::string, int>               sizes,
                      std::map<std::string, int>               Out_strides,
                      std::map<std::string, int>               In_strides)
        : loop_tree_program(
              loop_nest_transpose_to_tree(order, Out_strides, In_strides),
              sizes)
    {
    }

    loop_tree_fn_type get_fn() const
    {
        std::vector<loop_tree_fn_type> sub_functions;

        for (auto const& c : this->nodes)
        {
            sub_functions.push_back(c->get_fn(sizes));
        }

        return [sub_functions](std::map<std::string, float*> const& tensors,
                               std::map<std::string, int> const& alphas) {
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
