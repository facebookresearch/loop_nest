// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/loop_tree/node.hpp"

namespace dabun
{
namespace loop_tree
{

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

        return [to_advance](std::vector<Arithmetic*>& tensors, int delta = 1)
        {
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

        return [to_adjust](std::vector<int>& alpha_offsets, int adjustment)
        {
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
           std::map<std::string, int> const& outer_iteration_depths,
           formulas_map_type const& formulas, bool debug_mode) const override
    {
        auto var      = this->var;
        auto delta    = this->delta;
        auto children = this->get_children();
        auto limit    = sizes.at(var);

        auto const [full, rest] = full_rest(limit, delta);

        report_vector report = {
            std::make_shared<node_report>(for_loop_node_info{
                1, 1, var, full + (rest ? 1 : 0), delta, limit})};

        std::vector<loop_tree_fn_type<Arithmetic>> full_fns, tail_fns;

        auto iteration_depths = outer_iteration_depths;

        int last_iteration = full + (rest ? 1 : 0) - 1;
        strong_assert(last_iteration >= 0);

        iteration_depths[var] += last_iteration;

        for (auto c : children)
        {
            auto inner_sizes = sizes;

            if (full)
            {
                inner_sizes[var] = delta;
                auto [fn, rep] =
                    c->get_fn(tensors_idx, inner_sizes, iteration_depths,
                              formulas, debug_mode);
                full_fns.push_back(fn);
                report[0]->children.insert(report[0]->children.end(),
                                           rep.begin(), rep.end());
            }
            if (rest)
            {
                inner_sizes[var] = rest;
                auto [fn, rep] =
                    c->get_fn(tensors_idx, inner_sizes, iteration_depths,
                              formulas, debug_mode);
                tail_fns.push_back(fn);
                report.insert(report.end(), rep.begin(), rep.end());
            }
        }

        auto tensor_advancer =
            get_tensor_advancer(tensors_idx, get_tensors_used());
        auto alpha_offsets_adjuster = get_alpha_offsets_adjuster(
            tensors_idx, get_output_tensors(), formulas);

        LN_LOG(DEBUG) << "loop_tree: Executing interpreted for(" << var << ","
                      << delta << ")\n";

        return {[full, full_fns, tensor_advancer, alpha_offsets_adjuster,
                 tail_fns](std::vector<Arithmetic*>& tensors,
                           std::vector<int>&         alpha_offsets)
                {
                    for (int i = 0; i < full; ++i)
                    {
                        for (auto const& fn : full_fns)
                        {
                            fn(tensors, alpha_offsets);
                        }
                        tensor_advancer(tensors, 1);
                        alpha_offsets_adjuster(alpha_offsets, 1);
                    }

                    for (auto const& fn : tail_fns)
                    {
                        fn(tensors, alpha_offsets);
                    }

                    tensor_advancer(tensors, -full);
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

} // namespace loop_tree
} // namespace dabun
