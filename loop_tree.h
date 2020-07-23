// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "log.h"
#include "loop_nest.h"

namespace facebook
{
namespace sysml
{
namespace aot
{

enum class node_kind
{
    for_type,
    compute_type,
    transpose_type,
    compute_jitter_type,
    transpose_jitter_type
};

// TODO(j): is there a better way?
// I put ops into enum to avoid passing yet 2 more
// templates throughtout (which also seemed to make
// downcasting tree nodes hard?)
enum class arithmetic_op_kind
{
    plus,
    multiplies,
    // TODO: need to add rest here
};

template <class ISA>
class ComputeNode;

template <class ISA>
class jitted_loop_nest_node;

template <class ISA>
class TransposeNode;

template <class ISA>
class for_loop_node;

using LoopTreeFunction =
    std::function<void(std::map<std::string, float*> const&)>;

template <class ISA>
class ProgramNode
{
private:
    node_kind                                      kind_;
    std::vector<std::shared_ptr<ProgramNode<ISA>>> children_;

public:
    virtual ~ProgramNode(){};

    explicit ProgramNode(node_kind kind)
        : kind_(kind)
    {
    }

    std::vector<std::shared_ptr<ProgramNode<ISA>>> const& get_children() const
    {
        return children_;
    }

    void
    set_children(std::vector<std::shared_ptr<ProgramNode<ISA>>> const& children)
    {
        children_ = children;
    }

    void set_children(std::vector<std::shared_ptr<ProgramNode<ISA>>>&& children)
    {
        children_ = std::move(children);
    }

    node_kind get_type() const { return kind_; }

    virtual LoopTreeFunction
    get_fn(std::map<std::string, int> const&) const = 0;

    virtual std::vector<std::string> get_tensors_used() const = 0;
    virtual std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const = 0;
    // virtual void set_limits(std::map<std::string, std::vector<int>> const&) =
    // 0;
};

template <class ISA>
class ComputeNode : public ProgramNode<ISA>
{
private:
    using super_type = ProgramNode<ISA>;

    // 0 -> A, 1 -> B, rest are followed tensors
    std::vector<std::string> inputs;

    std::string                                       output;
    std::map<std::string, std::map<std::string, int>> strides;
    arithmetic_op_kind                                plus;
    arithmetic_op_kind                                multiplies;
    // TODO(j): need to add elementwise ops here...

    friend class jitted_loop_nest_node<ISA>;

public:
    ComputeNode(std::vector<std::string> inputs, std::string output,
                std::map<std::string, std::map<std::string, int>> strides)
        : super_type(node_kind::compute_type)
        , inputs(inputs)
        , output(output)
        , strides(strides)
    {
    }

    std::vector<std::string> get_tensors_used() const
    {
        auto tensors_used = inputs;
        tensors_used.push_back(output);
        return tensors_used;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    // void set_limits(std::map<std::string, std::vector<int>> const&) override
    // {
    //     // do nothing...
    // }

    LoopTreeFunction get_fn(std::map<std::string, int> const&) const
    {
        return [inputs = this->inputs, output = this->output](
                   std::map<std::string, float*> const& tensors) {
            LN_LOG(DEBUG) << "Hit compute\n";
            assert(tensors.count(inputs[0]) && tensors.count(inputs[1]) &&
                   tensors.count(output));

            float* A = tensors.at(inputs[0]);
            float* B = tensors.at(inputs[1]);
            float* C = tensors.at(output);
            // TODO(j): generalize to other ops supported....
            LN_LOG(DEBUG) << "(A:" << A[0] << ") * (B:" << B[0] << ")"
                          << "\n";
            C[0] += A[0] * B[0];
        };
    }
};

template <class ISA>
class for_loop_node : public ProgramNode<ISA>
{
private:
    using super_type = ProgramNode<ISA>;

    std::string var;
    int         delta;

    std::vector<std::string>                          in_scope_tensor_names;
    std::map<std::string, std::map<std::string, int>> in_scope_tensor_strides;
    //    std::map<std::string, std::vector<int>>           limits;

    template <class R>
    friend class jitted_loop_nest_node;

private:
    void set_in_scope_tensor_info()
    {
        for (auto c : this->get_children())
        {
            auto node_tensor_names = c->get_tensors_used();
            in_scope_tensor_names.insert(in_scope_tensor_names.end(),
                                         node_tensor_names.begin(),
                                         node_tensor_names.end());

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

public:
    std::string const& get_var() const { return var; }
    int                get_delta() const { return delta; }

    for_loop_node(std::string var, int delta,
                  std::vector<std::shared_ptr<ProgramNode<ISA>>> children)
        : super_type(node_kind::for_type)
        , var(var)
        , delta(delta)
    {
        // LN_LOG(DEBUG)  << "For(" << var << "," << delta << ")" << "\n";
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

    // void
    // set_limits(std::map<std::string, std::vector<int>> const& limits)
    // override
    // {
    //     this->limits = limits;
    // }

    LoopTreeFunction get_fn(std::map<std::string, int> const& sizes) const
    {
        // TODO(j): issues with using stateful method in lambda...we don't
        // want that

        auto var      = this->var;
        auto delta    = this->delta;
        auto children = this->get_children();
        auto limit    = sizes.at(var);

        int full = limit / delta;
        LN_LOG(DEBUG) << "Full: " << full << "\n";
        int rest = limit % delta;

        std::vector<LoopTreeFunction> full_fns, tail_fns;

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
                auto s = sizes;
                s[var] = rest;
                tail_fns.push_back(c->get_fn(s));
            }
        }

        auto advancer = get_tensor_advancer(get_tensors_used());

        return [=](std::map<std::string, float*> const& tensors) {
            auto ts = tensors;
            for (int i = 0; i < full; ++i)
            {
                for (auto const& fn : full_fns)
                {
                    fn(ts);
                }
                advancer(ts);
            }
            for (auto const& fn : tail_fns)
            {
                fn(ts);
            }
        };
    }
};

// template <class ISA, class PlusType, class MultipliesType>
// std::shared_ptr<ProgramNode<ISA>>
// merge_loop_into_jitter(std::shared_ptr<ProgramNode<ISA>> node,
//                        TransposeNode<ISA, PlusType, MultipliesType>*
//                        child)
// {
//     // create initial transpose jitter and add in {node.var, node.delta}
//     to
//     // order in jitter
// }

// template <class ISA>
// std::shared_ptr<ProgramNode<ISA>>
// merge_loop_into_jitter(std::shared_ptr<ProgramNode<ISA>>         node,
//                                          TransposeJitterNode<ISA>* child)
// {
//     // add in {node.var, node.delta} to order in jitter
// }

template <class ISA>
class jitted_loop_nest_node : public ProgramNode<ISA>
{
private:
    using super_type = ProgramNode<ISA>;

    std::vector<std::string>                          inputs;
    std::string                                       output;
    std::vector<std::pair<std::string, int>>          order;
    std::map<std::string, int>                        sizes;
    std::map<std::string, std::set<std::string>>      formulas;
    std::map<std::string, std::map<std::string, int>> strides;
    arithmetic_op_kind                                plus;
    arithmetic_op_kind                                multiplies;

    std::vector<std::string> tensors_used;

    // TODO (j) how should we handle general type? should that be a template?
    // facebook::sysml::aot::unique_aot_fn
    facebook::sysml::aot::shared_aot_fn<void(float*, const float*, const float*,
                                             int)>
        compiled_fn;

public:
    jitted_loop_nest_node(
        std::vector<std::string> inputs, std::string output,
        std::vector<std::pair<std::string, int>>          order,
        std::map<std::string, int>                        sizes,
        std::map<std::string, std::set<std::string>>      formulas,
        std::map<std::string, std::map<std::string, int>> strides,
        arithmetic_op_kind plus, arithmetic_op_kind multiplies)
        : super_type(node_kind::compute_jitter_type)
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
        : super_type(node_kind::compute_jitter_type)
        , inputs(compute_jitter_node->inputs)
        , output(compute_jitter_node->output)
        , order(compute_jitter_node->order)
        , sizes(compute_jitter_node->sizes)
        , formulas(compute_jitter_node->formulas)
        , strides(compute_jitter_node->strides)
        , plus(compute_jitter_node->plus)
        , multiplies(compute_jitter_node->multiplies)
    {
        order.insert(order.begin(), {for_node->var, for_node->delta});
    }

    jitted_loop_nest_node(std::shared_ptr<for_loop_node<ISA>> for_node,
                          std::shared_ptr<ComputeNode<ISA>>   compute_node,
                          std::map<std::string, int>          sizes,
                          std::map<std::string, std::set<std::string>> formulas)
        : super_type(node_kind::compute_jitter_type)
        , inputs(compute_node->inputs)
        , output(compute_node->output)
        , order({{for_node->var, for_node->delta}})
        , sizes(sizes)
        , formulas(formulas)
        , strides(compute_node->strides)
        , plus(compute_node->plus)
        , multiplies(compute_node->multiplies)
    {
    }

    LoopTreeFunction get_fn(std::map<std::string, int> const&) const
    {
        auto jit_fn = facebook::sysml::aot::FMA_loop_nest_jitter<ISA>(
                          order, sizes, formulas.at(output),
                          formulas.at(inputs[0]), formulas.at(inputs[1]),
                          strides.at(output), strides.at(inputs[0]),
                          strides.at(inputs[1]), facebook::sysml::aot::fma)
                          .get_shared();

        auto output = this->output;
        auto inputs = this->inputs;

        return [jit_fn, inputs, output](std::map<std::string, float*> tensors) {
            jit_fn(tensors.at(output), tensors.at(inputs[0]),
                   tensors.at(inputs[1]), 1);
        };
    }

    std::vector<std::string> get_tensors_used() const override
    {
        return tensors_used;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const override
    {
        return strides;
    }
};

template <class ISA>
std::shared_ptr<ProgramNode<ISA>>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>>         node,
                       std::shared_ptr<jitted_loop_nest_node<ISA>> child)
{
    return std::shared_ptr<ProgramNode<ISA>>(
        new jitted_loop_nest_node<ISA>(node, child));
}

template <class ISA>
std::shared_ptr<ProgramNode<ISA>>
merge_loop_into_jitter(std::shared_ptr<for_loop_node<ISA>>          node,
                       std::shared_ptr<ComputeNode<ISA>>            child,
                       std::map<std::string, int>                   sizes,
                       std::map<std::string, std::set<std::string>> formulas)
{
    return std::shared_ptr<ProgramNode<ISA>>(
        new jitted_loop_nest_node<ISA>(node, child, sizes, formulas));
}

template <class ISA>
std::shared_ptr<ProgramNode<ISA>>
simplify_loop_nests(std::shared_ptr<ProgramNode<ISA>>            node,
                    std::map<std::string, int>                   sizes,
                    std::map<std::string, std::set<std::string>> formulas)
{
    if (node->get_type() != node_kind::for_type)
    {
        return node;
    }

    std::vector<std::shared_ptr<ProgramNode<ISA>>> new_children;
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
    std::shared_ptr<ProgramNode<ISA>> single_child = new_children.at(0);

    switch (single_child->get_type())
    {
    case node_kind::compute_type:
        return merge_loop_into_jitter(
            for_node, std::dynamic_pointer_cast<ComputeNode<ISA>>(single_child),
            sizes, formulas);
        break;

    case node_kind::compute_jitter_type:
        return merge_loop_into_jitter(
            for_node, std::dynamic_pointer_cast<jitted_loop_nest_node<ISA>>(
                          single_child));
        break;

    case node_kind::transpose_type:
        // TODO(j): need to merge into jitter
        assert("Unhandled merger" && false);
        return node;
        break;
    default:
        assert("Unhandled merger" && false);
        return node;
    }
}

template <class ISA>
class Program
{
private:
    std::vector<std::shared_ptr<ProgramNode<ISA>>> nodes;
    std::map<std::string, int>                     sizes;
    std::map<std::string, std::set<std::string>>   formulas;

    static std::vector<std::shared_ptr<ProgramNode<ISA>>>
    loop_nest_compute_to_tree(std::vector<std::pair<std::string, int>> order,
                              std::map<std::string, int> C_strides,
                              std::map<std::string, int> A_strides,
                              std::map<std::string, int> B_strides)
    {
        auto innermost = std::shared_ptr<ComputeNode<ISA>>(new ComputeNode<ISA>(
            {"A", "B"}, "C",
            {{"A", A_strides}, {"B", B_strides}, {"C", C_strides}}));

        std::shared_ptr<ProgramNode<ISA>> current = innermost;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            auto new_node = std::shared_ptr<for_loop_node<ISA>>(
                new for_loop_node<ISA>(it->first, it->second, {current}));
            current = new_node;
        }

        return {current};
    }

    // std::vector<std::shared_ptr<ProgramNode<ISA>>>
    // loop_nest_transpose_to_tree(std::vector<std::pair<std::string, int>>
    // order,
    //                             std::map<std::string, int> C_strides,
    //                             std::map<std::string, int> A_strides)
    // {
    //     TransposeNode<ISA>* innermost = new TransposeNode<ISA>(
    //         "A", "C", {{"A", A_strides}, {"C", C_strides}});

    //     std::shared_ptr<ProgramNode<ISA>> current = innermost;
    //     for (auto it = order.rbegin(); it != order.rend(); it++)
    //     {
    //         std::shared_ptr<ProgramNode<ISA>> new_node =
    //             new for_loop_node<ISA>(it->first, it->second, {current});
    //         current = new_node;
    //     }

    //     return {current};
    // }

    // std::map<std::string, std::vector<int>>
    // sizes_to_limits(std::map<std::string, int> sizes)
    // {
    //     std::map<std::string, std::vector<int>> limits;
    //     for (auto const& s : sizes)
    //     {
    //         limits[s.first].push_back(s.second);
    //     }
    //     return limits;
    // }

public:
    Program(std::vector<std::shared_ptr<ProgramNode<ISA>>> nodes,
            std::map<std::string, int>                     sizes,
            std::map<std::string, std::set<std::string>>   formulas)
        : nodes(nodes)
        , sizes(sizes)
        , formulas(formulas)
    {
        // int  i      = 0;
        // // auto limits = sizes_to_limits(sizes);
        // LN_LOG(DEBUG) << "Original limits size:" << limits.size() << "\n";
        // for (auto c : nodes)
        // {
        //     LN_LOG(DEBUG) << "i: " << i << "\n";
        //     // c->set_limits(limits);
        //     i++;
        // }
        // LN_LOG(DEBUG) << "Size:" << nodes.size() << "\n";

        // PASS
        std::cout << "Simplifying loop nests" << std::endl;
        std::vector<std::shared_ptr<ProgramNode<ISA>>> new_nodes;
        for (auto c : nodes)
        {
            new_nodes.push_back(simplify_loop_nests(c, sizes, formulas));
        }
        nodes = new_nodes;

        // PASS
        // std::cout << "Compiling loop nests" << std::endl;
        // for (auto c : nodes)
        // {
        //     compile_loop_nests(c);
        // }
    }

    std::vector<std::shared_ptr<ProgramNode<ISA>>> get_children()
    {
        return nodes;
    }

    Program(std::vector<std::pair<std::string, int>> order,
            std::map<std::string, int> sizes, std::set<std::string> C_formula,
            std::set<std::string> A_formula, std::set<std::string> B_formula,
            std::map<std::string, int> C_strides,
            std::map<std::string, int> A_strides,
            std::map<std::string, int> B_strides)
        : Program(
              loop_nest_compute_to_tree(order, C_strides, A_strides, B_strides),
              sizes, {{"C", C_formula}, {"A", A_formula}, {"B", B_formula}})
    {
    }

    // Program(std::vector<std::pair<std::string, int>> order,
    //         std::map<std::string, int>               sizes,
    //         std::map<std::string, int>               Out_strides,
    //         std::map<std::string, int>               In_strides)
    //     : Program(loop_nest_transpose_to_tree(order, Out_strides,
    //     In_strides),
    //               sizes)
    // {
    // }

    LoopTreeFunction get_fn() const
    {
        auto nodes = this->nodes;

        std::vector<LoopTreeFunction> sub_functions;

        for (auto const& c : this->nodes)
        {
            sub_functions.push_back(c->get_fn(sizes));
        }

        return [sub_functions](std::map<std::string, float*> const& tensors) {
            for (auto f : sub_functions)
            {
                f(tensors);
            }
        };
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
