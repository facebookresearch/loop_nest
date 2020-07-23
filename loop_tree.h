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

enum class NodeType
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
enum class ArithmeticOps
{
    plus,
    multiplies,
    // TODO: need to add rest here
};

template <class ISA>
class ComputeNode;

template <class ISA>
class ComputeJitterNode;

template <class ISA>
class TransposeNode;

template <class ISA>
class ForNode;

using LoopTreeFunction = std::function<void(std::map<std::string, float*>)>;

template <class ISA>
class ProgramNode
{
public:
    virtual ~ProgramNode(){};
    virtual std::vector<ProgramNode<ISA>*> get_children() const           = 0;
    virtual void             set_children(std::vector<ProgramNode<ISA>*>) = 0;
    virtual LoopTreeFunction get_fn() const                               = 0;
    virtual NodeType         get_type() const                             = 0;

    virtual std::vector<std::string> const& get_tensors_used() const = 0;
    virtual std::map<std::string, std::map<std::string, int>> const&
                 get_tensor_strides() const                          = 0;
    virtual void set_limits(std::map<std::string, std::vector<int>>) = 0;
};

template <class ISA>
class ComputeNode : public ProgramNode<ISA>
{
private:
    std::vector<std::string>                          inputs;
    std::string                                       output;
    std::map<std::string, std::map<std::string, int>> strides;
    ArithmeticOps                                     plus;
    ArithmeticOps                                     multiplies;
    // TODO(j): need to add elementwise ops here...

    static const NodeType    type = NodeType::compute_type;
    std::vector<std::string> tensors_used;

    friend class ComputeJitterNode<ISA>;

public:
    ComputeNode(std::vector<std::string> inputs, std::string output,
                std::map<std::string, std::map<std::string, int>> strides)
        : inputs(inputs)
        , output(output)
        , strides(strides)
    {
        tensors_used = inputs;
        tensors_used.push_back(output);
    }

    std::vector<std::string> const& get_tensors_used() const
    {
        return tensors_used;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    void set_limits(std::map<std::string, std::vector<int>> limits)
    {
        // do nothing...
    }

    std::vector<ProgramNode<ISA>*> get_children() const { return {}; }
    void set_children(std::vector<ProgramNode<ISA>*>) {}

    NodeType get_type() const { return type; }

    LoopTreeFunction get_fn() const
    {
        return [this](std::map<std::string, float*> tensors) {
            auto inputs = this->inputs;
            auto output = this->output;
            LN_LOG(DEBUG) << "Hit compute\n";
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
class ComputeJitterNode : public ProgramNode<ISA>
{
private:
    std::vector<std::string>                          inputs;
    std::string                                       output;
    std::vector<std::pair<std::string, int>>          order;
    std::map<std::string, int>                        sizes;
    std::map<std::string, std::set<std::string>>      formulas;
    std::map<std::string, std::map<std::string, int>> strides;
    ArithmeticOps                                     plus;
    ArithmeticOps                                     multiplies;
    // TODO(j): need to add elementwise ops here...

    static const NodeType    type = NodeType::compute_jitter_type;
    std::vector<std::string> tensors_used;

    // TODO (j) how should we handle general type? should that be a template?
    // facebook::sysml::aot::unique_aot_fn
    facebook::sysml::aot::shared_aot_fn<void(float*, const float*, const float*,
                                             int)>
        compiled_fn;

public:
    ComputeJitterNode(std::vector<std::string> inputs, std::string output,
                      std::vector<std::pair<std::string, int>>     order,
                      std::map<std::string, int>                   sizes,
                      std::map<std::string, std::set<std::string>> formulas,
                      std::map<std::string, std::map<std::string, int>> strides,
                      ArithmeticOps plus, ArithmeticOps multiplies)
        : inputs(inputs)
        , output(output)
        , order(order)
        , sizes(sizes)
        , formulas(formulas)
        , strides(strides)
        , plus(plus)
        , multiplies(multiplies)
    {
    }

    ComputeJitterNode(ForNode<ISA>*           for_node,
                      ComputeJitterNode<ISA>* compute_jitter_node)
        : inputs(compute_jitter_node->inputs)
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

    ComputeJitterNode(ForNode<ISA>* for_node, ComputeNode<ISA>* compute_node,
                      std::map<std::string, int>                   sizes,
                      std::map<std::string, std::set<std::string>> formulas)
        : inputs(compute_node->inputs)
        , output(compute_node->output)
        , order({{for_node->var, for_node->delta}})
        , sizes(sizes)
        , formulas(formulas)
        , strides(compute_node->strides)
        , plus(compute_node->plus)
        , multiplies(compute_node->multiplies)
    {
    }

    std::vector<ProgramNode<ISA>*> get_children() const { return {}; }
    void set_children(std::vector<ProgramNode<ISA>*>) {}

    void jit_compile()
    {
        // TODO(j): need to put in appropriate op here
        this->compiled_fn =
            facebook::sysml::aot::FMA_loop_nest_jitter<ISA>(
                order, sizes, formulas.at(output), formulas.at(inputs[0]),
                formulas.at(inputs[1]), strides.at(output),
                strides.at(inputs[0]), strides.at(inputs[1]),
                facebook::sysml::aot::fma)
                .get_shared();
        this->compiled_fn.save_to_file("compiled.asm");
        // compiled_fn = fn;
        // facebook::sysml::aot::aot_fn_cast<void(
        //   float*, float const*, float const*, int)>(std::move(fn));
    }

    LoopTreeFunction get_fn() const
    {
        auto myfn = this->compiled_fn;
        myfn.save_to_file("before_jumping_in_lambda.asm");

        auto output   = this->output;
        auto inputs   = this->inputs;
        auto order    = this->order;
        auto formulas = this->formulas;
        auto sizes    = this->sizes;
        auto strides  = this->strides;

        return [myfn, output, inputs, order, formulas, sizes,
                strides](std::map<std::string, float*> tensors) {
            // this works (and is the same call I make when
            // building compiled_fn
            auto fn = facebook::sysml::aot::FMA_loop_nest_jitter<ISA>(
                          order, sizes, formulas.at(output),
                          formulas.at(inputs[0]), formulas.at(inputs[1]),
                          strides.at(output), strides.at(inputs[0]),
                          strides.at(inputs[1]), facebook::sysml::aot::fma)
                          .get_shared();
            fn(tensors.at(output), tensors.at(inputs[0]), tensors.at(inputs[1]),
               1);
            fn.save_to_file("fresh_compiled_in_lambda.asm");

            // TODO(j):
            // dump code out
            // assert tensors.
            // this segfaults
            myfn.save_to_file("reuse_compiled_in_lambda.asm");
            myfn(tensors.at(output), tensors.at(inputs[0]),
                 tensors.at(inputs[1]), 1);
        };
    }

    NodeType get_type() const { return type; }

    virtual std::vector<std::string> const& get_tensors_used() const
    {
        return tensors_used;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    void set_limits(std::map<std::string, std::vector<int>>)
    {
        // do nothing
    }
};

template <class ISA>
class TransposeNode : public ProgramNode<ISA>
{

private:
    std::string                                       input;
    std::string                                       output;
    std::map<std::string, std::map<std::string, int>> strides;

    static const NodeType    type = NodeType::transpose_type;
    std::vector<std::string> tensors_used;

public:
    TransposeNode(std::string input, std::string output,
                  std::map<std::string, std::map<std::string, int>> strides)
        : input(input)
        , output(output)
        , strides(strides)
    {
        tensors_used.push_back(input);
        tensors_used.push_back(output);
    }

    std::vector<std::string> const& get_tensors_used() const
    {
        return tensors_used;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    void set_limits(std::map<std::string, std::vector<int>>)
    {
        // do nothing...
    }

    std::vector<ProgramNode<ISA>*> get_children() const { return {}; }

    void set_children(std::vector<ProgramNode<ISA>*>) {}

    NodeType get_type() const { return type; }

    LoopTreeFunction get_fn() const
    {
        return [this](std::map<std::string, float*> tensors) {
            auto input  = this->input;
            auto output = this->output;
            LN_LOG(DEBUG) << "Hit transpose\n";
            float* A = tensors.at(input);
            float* C = tensors.at(output);
            // TODO(j): generalize to other ops supported....
            LN_LOG(DEBUG) << "(C:" << C[0] << ") * (A:" << A[0] << ")"
                          << "\n";
            C[0] = A[0];
        };
    }
};

template <class ISA>
class ForNode : public ProgramNode<ISA>
{
private:
    std::string                    var;
    int                            delta;
    std::vector<ProgramNode<ISA>*> children;

    std::vector<std::string>                          in_scope_tensor_names;
    std::map<std::string, std::map<std::string, int>> in_scope_tensor_strides;
    std::map<std::string, std::vector<int>>           limits;

    static const NodeType type = NodeType::for_type;

    template <class R>
    friend class ComputeJitterNode;

private:
    void set_in_scope_tensor_info()
    {
        for (auto c : children)
        {
            std::vector<std::string> node_tensor_names = c->get_tensors_used();
            in_scope_tensor_names.insert(in_scope_tensor_names.end(),
                                         node_tensor_names.begin(),
                                         node_tensor_names.end());

            std::map<std::string, std::map<std::string, int>>
                node_tensor_strides = c->get_tensor_strides();

            in_scope_tensor_strides.insert(node_tensor_strides.begin(),
                                           node_tensor_strides.end());
        }
    }

    void advance_tensors(std::vector<std::string>       tensor_names,
                         std::map<std::string, float*>& tensors) const
    {
        for (std::string name : tensor_names)
        {
            std::int64_t offset =
                in_scope_tensor_strides.at(name).count(var)
                    ? in_scope_tensor_strides.at(name).at(var) * delta
                    : 0;

            LN_LOG(DEBUG) << name << "+=" << offset << "\n";
            tensors[name] += offset;
        }
    }

public:
    ForNode(std::string var, int delta, std::vector<ProgramNode<ISA>*> children)
        : var(var)
        , delta(delta)
        , children(children)
    {
        // LN_LOG(DEBUG)  << "For(" << var << "," << delta << ")" << "\n";
        set_in_scope_tensor_info();
    }

    std::vector<ProgramNode<ISA>*> get_children() const { return children; }

    void set_children(std::vector<ProgramNode<ISA>*> new_children)
    {
        children = new_children;
    }

    NodeType get_type() const { return type; }

    std::vector<std::string> const& get_tensors_used() const
    {
        return in_scope_tensor_names;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return in_scope_tensor_strides;
    }

    void set_limits(std::map<std::string, std::vector<int>> limits)
    {
        this->limits = limits;
    }

    LoopTreeFunction get_fn() const
    {
        // TODO(j): issues with using stateful method in lambda...we don't want
        // that
        return [this](std::map<std::string, float*> tensors) {
            auto var      = this->var;
            auto delta    = this->delta;
            auto children = this->children;
            auto limits   = this->limits;

            // LN_LOG(DEBUG)  << "For(" << var << "," << delta << ")" <<
            // "\n"; LN_LOG(DEBUG)  << "Size limits:" << limits.size() <<
            // "\n";
            auto limit = limits[var].back();
            int  full  = limit / delta;
            LN_LOG(DEBUG) << "Full: " << full << "\n";
            int rest = limit % delta;

            for (auto c : children)
            {
                limits[var].push_back(delta);
                c->set_limits(limits);

                for (int i = 0; i < full; i++)
                {
                    c->get_fn()(tensors);
                    this->advance_tensors(c->get_tensors_used(), tensors);
                }

                limits[var].pop_back();
                c->set_limits(limits);

                if (rest)
                {
                    limits[var].push_back(rest);
                    c->set_limits(limits);

                    c->get_fn()(tensors);

                    limits[var].pop_back();
                    c->set_limits(limits);
                }
            }
        };
    }
};

template <class ISA>
ProgramNode<ISA>*
merge_loop_into_jitter(ForNode<ISA>* node, ComputeNode<ISA>* child,
                       std::map<std::string, int>                   sizes,
                       std::map<std::string, std::set<std::string>> formulas)
{
    return new ComputeJitterNode<ISA>(node, child, sizes, formulas);
}

template <class ISA>
ProgramNode<ISA>* merge_loop_into_jitter(ForNode<ISA>*           node,
                                         ComputeJitterNode<ISA>* child)
{
    return new ComputeJitterNode<ISA>(node, child);
}

// template <class ISA, class PlusType, class MultipliesType>
// ProgramNode<ISA>*
// merge_loop_into_jitter(ProgramNode<ISA>*                             node,
//                        TransposeNode<ISA, PlusType, MultipliesType>* child)
// {
//     // create initial transpose jitter and add in {node.var, node.delta} to
//     // order in jitter
// }

// template <class ISA>
// ProgramNode<ISA>* merge_loop_into_jitter(ProgramNode<ISA>*         node,
//                                          TransposeJitterNode<ISA>* child)
// {
//     // add in {node.var, node.delta} to order in jitter
// }

template <class ISA>
ProgramNode<ISA>*
simplify_loop_nests(ProgramNode<ISA>* node, std::map<std::string, int> sizes,
                    std::map<std::string, std::set<std::string>> formulas)
{
    if (node->get_type() != NodeType::for_type)
    {
        return node;
    }

    std::vector<ProgramNode<ISA>*> new_children;
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

    ForNode<ISA>*     for_node     = dynamic_cast<ForNode<ISA>*>(node);
    ProgramNode<ISA>* single_child = new_children.at(0);

    switch (single_child->get_type())
    {
    case NodeType::compute_type:
        return merge_loop_into_jitter(
            for_node, dynamic_cast<ComputeNode<ISA>*>(single_child), sizes,
            formulas);
        break;

    case NodeType::compute_jitter_type:
        return merge_loop_into_jitter(
            for_node, dynamic_cast<ComputeJitterNode<ISA>*>(single_child));
        break;

    case NodeType::transpose_type:
        // TODO(j): need to merge into jitter
        return node;
        break;
    default:
        assert("Unhandled merger" && false);
        return node;
    }
}

template <class ISA>
void compile_loop_nests(ProgramNode<ISA>* node)
{
    switch (node->get_type())
    {
    case NodeType::compute_jitter_type:
        dynamic_cast<ComputeJitterNode<ISA>*>(node)->jit_compile();
        break;
    case NodeType::transpose_jitter_type:
        // TODO(j): handle
        // static_cast<TransposeJitterNode *>(node)->jit_compile();
        break;
    case NodeType::for_type:
        for (auto child : node->get_children())
        {
            compile_loop_nests(child);
        }
        break;
    default:
        // do nothing
        break;
    }
}

template <class ISA>
class Program
{
private:
    std::vector<ProgramNode<ISA>*>               nodes;
    std::map<std::string, int>                   sizes;
    std::map<std::string, std::set<std::string>> formulas;

    std::vector<ProgramNode<ISA>*>
    loop_nest_compute_to_tree(std::vector<std::pair<std::string, int>> order,
                              std::map<std::string, int> C_strides,
                              std::map<std::string, int> A_strides,
                              std::map<std::string, int> B_strides)
    {
        ComputeNode<ISA>* innermost = new ComputeNode<ISA>(
            {"A", "B"}, "C",
            {{"A", A_strides}, {"B", B_strides}, {"C", C_strides}});

        ProgramNode<ISA>* current = innermost;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            ProgramNode<ISA>* new_node =
                new ForNode<ISA>(it->first, it->second, {current});
            current = new_node;
        }

        return {current};
    }

    std::vector<ProgramNode<ISA>*>
    loop_nest_transpose_to_tree(std::vector<std::pair<std::string, int>> order,
                                std::map<std::string, int> C_strides,
                                std::map<std::string, int> A_strides)
    {
        TransposeNode<ISA>* innermost = new TransposeNode<ISA>(
            "A", "C", {{"A", A_strides}, {"C", C_strides}});

        ProgramNode<ISA>* current = innermost;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            ProgramNode<ISA>* new_node =
                new ForNode<ISA>(it->first, it->second, {current});
            current = new_node;
        }

        return {current};
    }

    std::map<std::string, std::vector<int>>
    sizes_to_limits(std::map<std::string, int> sizes)
    {
        std::map<std::string, std::vector<int>> limits;
        for (auto const& s : sizes)
        {
            limits[s.first].push_back(s.second);
        }
        return limits;
    }

public:
    Program(std::vector<ProgramNode<ISA>*>               nodes,
            std::map<std::string, int>                   sizes,
            std::map<std::string, std::set<std::string>> formulas)
        : nodes(nodes)
        , sizes(sizes)
        , formulas(formulas)
    {
        int  i      = 0;
        auto limits = sizes_to_limits(sizes);
        LN_LOG(DEBUG) << "Original limits size:" << limits.size() << "\n";
        for (auto c : nodes)
        {
            LN_LOG(DEBUG) << "i: " << i << "\n";
            c->set_limits(limits);
            i++;
        }
        LN_LOG(DEBUG) << "Size:" << nodes.size() << "\n";

        // PASS
        std::cout << "Simplifying loop nests" << std::endl;
        std::vector<ProgramNode<ISA>*> new_nodes;
        for (auto c : nodes)
        {
            ProgramNode<ISA>* new_node =
                simplify_loop_nests(c, sizes, formulas);
            new_nodes.push_back(new_node);
        }
        nodes = new_nodes;

        // PASS
        std::cout << "Compiling loop nests" << std::endl;
        for (auto c : nodes)
        {
            compile_loop_nests(c);
        }
    }

    std::vector<ProgramNode<ISA>*> get_children() { return nodes; }

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

    Program(std::vector<std::pair<std::string, int>> order,
            std::map<std::string, int>               sizes,
            std::map<std::string, int>               Out_strides,
            std::map<std::string, int>               In_strides)
        : Program(loop_nest_transpose_to_tree(order, Out_strides, In_strides),
                  sizes)
    {
    }

    LoopTreeFunction get_fn() const
    {
        auto nodes = this->nodes;

        return [this](std::map<std::string, float*> tensors) {
            auto nodes = this->nodes;
            for (auto c : nodes)
            {
                c->get_fn()(tensors);
            }
        };
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
