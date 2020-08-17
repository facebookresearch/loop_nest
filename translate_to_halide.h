// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

#include "Halide.h"

#include "baselines.h"
#include "isa.h"
#include "utils.h"

namespace facebook
{
namespace sysml
{
namespace aot
{

// post-ops
class halide_elementwise_op
{
};
class halide_relu_op : public halide_elementwise_op
{
};
inline std::shared_ptr<halide_elementwise_op> const halide_relu =
    std::make_shared<halide_relu_op>();

template <class ISA>
class LoopNestToHalide
{
private:
    // loop_nest inputs
    std::shared_ptr<halide_elementwise_op>   elementwise;
    std::vector<std::pair<std::string, int>> order;
    std::map<std::string, int>               sizes;

    std::set<std::string> C_formula;
    std::set<std::string> A_formula;
    std::set<std::string> B_formula;

    std::map<std::string, int> C_strides;
    std::map<std::string, int> A_strides;
    std::map<std::string, int> B_strides;

    int                  max_fmas_unrolled;
    static constexpr int vector_size =
        facebook::sysml::aot::isa_traits<ISA>::vector_size;
    static constexpr int alignment = vector_size * 4;

    // Halide info
    std::int64_t A_size;
    std::int64_t B_size;
    std::int64_t C_size;

    Halide::Buffer<float> A_param;
    Halide::Buffer<float> B_param;
    std::vector<int>      C_dim_sizes;
    Halide::Target halide_target = Halide::get_jit_target_from_environment();
    Halide::Func   halide_program;

    Halide::RDom halide_rdom;
    // maps reduction dimension name to position in RDom
    std::map<std::string, int> reduction_dims_position;

    // keeps track of all vars/rvars throughout the program
    // mapping var/rvar name to object
    std::map<std::string, Halide::Var>  halide_vars;
    std::map<std::string, Halide::RVar> halide_rvars;

    // keeps track of the var/rvar associated with a
    // dimension as we iterate over order input
    std::map<std::string, Halide::Var>  halide_current_vars;
    std::map<std::string, Halide::RVar> halide_current_rvars;

    // maps from variable name to extent/stride
    std::map<std::string, int> halide_extents;
    std::map<std::string, int> halide_strides;

    // order of variables for iteration
    std::vector<std::string> halide_reorder;

    std::stringstream halide_code;

private:
    // utility
    template <class K, class V>
    std::vector<std::pair<K, V>> map_to_vector(std::map<K, V> m) const
    {
        std::vector<std::pair<K, V>> vec;
        for (auto const& entry : m)
        {
            vec.push_back({entry.first, entry.second});
        }
        auto compare_values = [](auto a, auto b) {
            return a.second < b.second;
        };
        std::sort(vec.begin(), vec.end(), compare_values);
        return vec;
    }

    Halide::Buffer<float> create_buffer(const std::string& name,
                                        std::int64_t       size)
    {
        // we create buffer with single dimension (flattened out)
        Halide::Buffer tensor(Halide::type_of<float>(), size, name);
        halide_code << "Buffer " << name << "(Float(32), " << size << ", \""
                    << name << "\");\n";
        return tensor;
    }

    Halide::RVar create_rvar(const Halide::RDom& rdom,
                             const std::string dim_name, int i, int extent)
    {
        Halide::RVar dim_rvar          = rdom[i];
        std::string  var_name          = dim_rvar.name();
        halide_rvars[var_name]         = dim_rvar;
        halide_extents[var_name]       = extent;
        halide_strides[var_name]       = 1;
        halide_current_rvars[dim_name] = dim_rvar;

        halide_code << "//" << dim_name << " points to rdom[" << i << "]\n";

        return dim_rvar;
    }

    Halide::RVar create_rvar(const std::string dim_name,
                             const std::string var_name, int extent)
    {
        Halide::RVar rvar(var_name);
        halide_rvars[var_name]         = rvar;
        halide_extents[var_name]       = extent;
        halide_strides[var_name]       = 1;
        halide_current_rvars[dim_name] = rvar;

        halide_code << "RVar " << var_name << "(\"" << var_name << "\");\n";

        return rvar;
    }

    Halide::Var create_var(const std::string dim_name,
                           const std::string var_name, int extent)
    {
        Halide::Var var(var_name);
        halide_vars[var_name]         = var;
        halide_extents[var_name]      = extent;
        halide_strides[var_name]      = 1;
        halide_current_vars[dim_name] = var;

        halide_code << "Var " << var_name << "(\"" << var_name << "\"); \n";

        return var;
    }

    Halide::VarOrRVar create_var_or_rvar(const std::string dim_name,
                                         const std::string var_name, int extent)
    {
        return is_reduction_dim(dim_name)
                   ? Halide::VarOrRVar(create_rvar(dim_name, var_name, extent))
                   : Halide::VarOrRVar(create_var(dim_name, var_name, extent));
    }

    Halide::VarOrRVar create_child_var_or_rvar(const std::string dim_name,
                                               const Halide::VarOrRVar& parent,
                                               int extent = 0)
    {
        if (extent == 0)
        {
            // share same extent as parent
            extent = halide_extents[parent.name()];
        }

        std::string new_name = parent.name();
        new_name.append("_i");
        Halide::VarOrRVar child_var =
            create_var_or_rvar(dim_name, new_name, extent);
        return child_var;
    }

    Halide::VarOrRVar get_current_dim_obj(const std::string& dim) const
    {
        return is_reduction_dim(dim)
                   ? Halide::VarOrRVar(halide_current_rvars.at(dim))
                   : Halide::VarOrRVar(halide_current_vars.at(dim));
    }

    Halide::VarOrRVar get_var_or_rvar_obj(const std::string& var_name) const
    {
        return halide_rvars.count(var_name)
                   ? Halide::VarOrRVar(halide_rvars.at(var_name))
                   : Halide::VarOrRVar(halide_vars.at(var_name));
    }

    int get_dimension_size(const std::string& dim) const
    {
        return sizes.at(dim);
    }

    int get_extent(const std::string& name) const
    {
        return halide_extents.at(name);
    }

    int get_stride(const std::string& name) const
    {
        return halide_strides.at(name);
    }

    void create_initial_vars_and_rvars()
    {
        for (auto const& entry : sizes)
        {
            std::string dim_name = entry.first;
            if (C_strides.count(dim_name) || A_strides.count(dim_name) ||
                B_strides.count(dim_name))
            {
                if (is_reduction_dim(dim_name))
                {
                    create_rvar(halide_rdom, dim_name,
                                reduction_dims_position[dim_name],
                                get_dimension_size(dim_name));
                }
                else
                {
                    create_var(dim_name, dim_name,
                               get_dimension_size(dim_name));
                }
            }
        }
    }

    void set_reduction_dims()
    {
        for (auto step = order.rbegin(); step != order.rend(); step++)
        {
            if ((C_formula.count(step->first) == 0) &&
                (reduction_dims_position.count(step->first) == 0))
            {
                reduction_dims_position[step->first] =
                    reduction_dims_position.size();
            }
        }

        // degenerate schedules can have
        // reduction dimensions not mentioned in the order
        // in general this doens't happen, but here for completeness
        for (auto const& e : A_strides)
        {
            if ((C_formula.count(e.first) == 0) &&
                (reduction_dims_position.count(e.first) == 0))
            {
                reduction_dims_position[e.first] =
                    reduction_dims_position.size();
            }
        }
        for (auto const& e : B_strides)
        {
            if ((C_formula.count(e.first) == 0) &&
                (reduction_dims_position.count(e.first) == 0))
            {
                reduction_dims_position[e.first] =
                    reduction_dims_position.size();
            }
        }
    }

    bool is_reduction_dim(const std::string& dim) const
    {
        return reduction_dims_position.count(dim) > 0;
    }

    void create_rdom()
    {
        std::vector<Halide::Range>                      ranges;
        const std::vector<std::pair<std::string, int>>& reduction_dims =
            map_to_vector<std::string, int>(reduction_dims_position);

        std::stringstream ranges_str;
        int               n = reduction_dims.size();
        for (auto const& rdim : reduction_dims)
        {
            int           extent = get_dimension_size(rdim.first);
            Halide::Range r(0, extent);
            ranges.push_back(r);

            ranges_str << "{0, " << extent << "}";
            if (--n > 0)
            {
                ranges_str << ",";
            }
        }

        halide_rdom = Halide::RDom(ranges, "r_dom");
        halide_code << "RDom r_dom({" << ranges_str.str() << "}, \"r_dom\");\n";
    }

    Halide::Func bound_initial_vars(Halide::Func C)
    {
        for (auto const& entry : sizes)
        {
            std::string dim_name = entry.first;
            // reduction dimensions are bounded directly
            // in the RDom
            if ((!is_reduction_dim(dim_name)) &&
                (C_strides.count(dim_name) || A_strides.count(dim_name) ||
                 B_strides.count(dim_name)))
            {
                int         size = get_dimension_size(dim_name);
                Halide::Var var  = get_var_or_rvar_obj(dim_name).var;
                C.bound(var, 0, size);

                halide_code << C.name() << ".bound(" << var.name() << ",0,"
                            << size << ");\n";
            }
        }
        std::vector<std::pair<std::string, int>> C_s;

        for (auto it : C_strides)
        {
            C_s.push_back({it.first, it.second});
        }
        std::sort(
            C_s.begin(), C_s.end(),
            [](std::pair<std::string, int> a, std::pair<std::string, int> b) {
                return a.second < b.second;
            });

        for (int i = 0; i < C.dimensions(); ++i)
        {
            std::string dim_name = C_s.at(i).first;

            C.output_buffer().dim(i).set_min(0);
            C.output_buffer().dim(i).set_stride(C_strides.at(dim_name));
            if (i + 1 < C_s.size())
            {
                std::string next_dim_name = C_s.at(i + 1).first;
                int extent     = C_strides[next_dim_name] / C_strides[dim_name];
                int max_extent = sizes.at(dim_name);
                // Occasionally we can have loop_nest inputs that
                // state a stride > size of a dimension
                // loop_nest handles this appropriately and simply never
                // advances... however, for halide, we need to
                // set the extent to 1 and the stride of the following output
                // dimension in C to 1, to avoid bound constraint errors
                if (extent > max_extent)
                {
                    extent                   = max_extent;
                    C_strides[next_dim_name] = 1;
                }
                C.output_buffer().dim(i).set_extent(extent);
            }
        }

        return C;
    }

    std::vector<std::string> get_C_dim_names() const
    {
        const std::vector<std::pair<std::string, int>>& strides_as_vec =
            map_to_vector<std::string, int>(C_strides);
        std::vector<std::string> dim_names;
        for (auto const& entry : strides_as_vec)
        {
            dim_names.push_back(entry.first);
        }
        return dim_names;
    }

    void set_C_dim_sizes()
    {
        const std::vector<std::string>& dim_names = get_C_dim_names();
        std::vector<int>                dim_sizes;

        std::int64_t total_size = 1;
        for (auto const& name : dim_names)
        {
            int ds = get_dimension_size(name);
            dim_sizes.push_back(ds);
            total_size *= ds;
        }

        C_dim_sizes = dim_sizes;
        C_size      = total_size;
    }

    Halide::Expr flat_buffer_access(Halide::Buffer<float>&            buf,
                                    const std::map<std::string, int>& strides,
                                    std::string                       name)
    {
        Halide::Expr      index = 0;
        std::stringstream index_str;
        index_str << "0";

        for (auto const& entry : strides)
        {
            Halide::VarOrRVar elem = get_current_dim_obj(entry.first);
            if (entry.second == 0)
            {
                continue;
            }

            if (is_reduction_dim(entry.first))
            {
                index += elem.rvar * entry.second;
            }
            else
            {
                index += elem.var * entry.second;
            }
            index_str << " + " << elem.name() << " * " << entry.second;
        }

        halide_code << name << "(" << index_str.str() << ")";

        return buf(index);
    }

    Halide::Func define_fma()
    {
        Halide::Func C("C");

        halide_code << "Func C(\"C\");\n";

        std::vector<std::string> C_dims_names = get_C_dim_names();
        std::vector<Halide::Var> C_dims;
        std::stringstream        C_dims_str;

        int n = C_dims_names.size();

        for (auto const& d : C_dims_names)
        {
            // be definition not an rvar
            C_dims.push_back(get_current_dim_obj(d).var);

            C_dims_str << d;

            if (--n > 0)
            {
                C_dims_str << ",";
            }
        }

        C(C_dims) = 0.0f;
        halide_code << "C(" << C_dims_str.str() << ") = 0.0f;\n";

        halide_code << "C(" << C_dims_str.str() << ") += ";
        Halide::Expr A_expr = flat_buffer_access(A_param, A_strides, "A");
        halide_code << " * ";
        Halide::Expr B_expr = flat_buffer_access(B_param, B_strides, "B");
        halide_code << ";\n";

        C(C_dims) += A_expr * B_expr;

        if (elementwise == facebook::sysml::aot::halide_relu)
        {
            C(C_dims) = Halide::max(0.0f, C(C_dims));
            halide_code << "C(" << C_dims_str.str() << ") = max(0.0f, C("
                        << C_dims_str.str() << "));\n";
        }
        return C;
    }

    std::pair<Halide::VarOrRVar, Halide::VarOrRVar>
    split(Halide::Func& C, const std::string dim, int stride)
    {
        Halide::VarOrRVar parent_var = get_current_dim_obj(dim);
        Halide::VarOrRVar inner_var =
            create_child_var_or_rvar(dim, parent_var, stride);
        halide_strides[parent_var.name()] = stride;

        Halide::TailStrategy tail = Halide::TailStrategy::GuardWithIf;

        C.update(0).split(parent_var, parent_var, inner_var, stride, tail);

        halide_code << C.name() << ".update(0).split(" << parent_var.name()
                    << "," << parent_var.name() << "," << inner_var.name()
                    << "," << std::to_string(stride)
                    << ", TailStrategy::GuardWithIf);\n";
        return std::make_pair(parent_var, inner_var);
    }

    Halide::Func unroll(Halide::Func& C)
    {
        int                            instructions_unrolled = 1;
        std::vector<Halide::VarOrRVar> possible_unrolls = get_reorder_vars();

        int ix = 0;
        for (auto it = possible_unrolls.begin(); it != possible_unrolls.end();
             it++)
        {
            // skip first one (that is the vectorized dimension)
            if (ix++ == 0)
                continue;
            float extent  = get_extent(it->name());
            float stride  = get_stride(it->name());
            int   unrolls = std::ceil(extent / stride);
            instructions_unrolled *= unrolls;

            if (instructions_unrolled <= max_fmas_unrolled)
            {
                C.update(0).unroll(*it);

                halide_code << C.name() << ".update(0).unroll(" << it->name()
                            << ");\n";
            }
            else
            {
                break;
            }
        }
        return C;
    }

    Halide::Func vectorize(Halide::Func& C)
    {
        auto innermost_step = order.rbegin();
        // we split in order to vectorize the right number of elements
        auto         pair = split(C, innermost_step->first, vector_size);
        Halide::Func current_func = C;

        // trying to vectorize over a RVar direclty will result in
        // an error since there are dependencies, need to factor out (`rfactor`)
        if (is_reduction_dim(innermost_step->first))
        {
            Halide::Var helper_var = create_var("helper_var", "helper_var",
                                                get_extent(pair.second.name()));
            add_to_reorder(helper_var.name());

            Halide::RVar reduction_var = pair.second.rvar;
            Halide::Func intermediate =
                current_func.update(0).rfactor({{reduction_var, helper_var}});
            intermediate.vectorize(helper_var);

            halide_code << "Var helper_var;\n Func intermediate = "
                        << current_func.name() << ".update(0).rfactor("
                        << reduction_var.name() << ", helper_var)\n";
            halide_code << "intermediate.vectorize(helper_var);\n";

            // we compute this intermediate in the prior loop level
            // when there are multiple loops
            if (order.size() > 1)
            {
                Halide::VarOrRVar prior_obj = get_current_dim_obj(innermost_step->first);

                // .compute_at does not overload on VarOrRVar so explicit
                if (prior_obj.is_rvar)
                {
                    intermediate.compute_at(current_func, prior_obj.rvar);
                }
                else
                {
                    intermediate.compute_at(current_func, prior_obj.var);
                }
                halide_code << "intermediate.compute_at(" << current_func.name()
                            << ", " << prior_obj.name() << ")\n";
            }
            else
            {
                intermediate.compute_at(current_func, pair.second.rvar);
                halide_code << "intermediate.compute_at(" << current_func.name()
                            << ", " << pair.second.name() << ")\n";
            }
            current_func = intermediate;
        }
        else
        {
            current_func.update(0).vectorize(pair.second);
            halide_code << current_func.name() << ".update(0).vectorize("
                        << pair.second.name() << ");\n";
            add_to_reorder(pair.second.name());
        }

        return current_func;
    }

    std::vector<Halide::VarOrRVar> get_reorder_vars() const
    {
        std::vector<Halide::VarOrRVar> reorder_vars;

        if (halide_reorder.size() == 0)
        {
            return reorder_vars;
        }

        // satisfy Halide innermost first
        std::map<std::string, int> already_added;
        for (auto it = halide_reorder.rbegin(); it != halide_reorder.rend();
             it++)
        {
            std::string var_name = *it;
            if (already_added.count(var_name) == 0)
            {
                reorder_vars.push_back(get_var_or_rvar_obj(var_name));
                already_added[var_name] = 1;
            }
        }

        return reorder_vars;
    }

    Halide::Func reorder(Halide::Func& C)
    {
        std::vector<Halide::VarOrRVar> vars = get_reorder_vars();
        C.update(0).reorder(vars);

        std::stringstream vars_str;
        int               n = vars.size();

        for (auto const& v : vars)
        {
            vars_str << v.name();
            if (--n > 0)
            {
                vars_str << ",";
            }
        }

        halide_code << C.name() << ".update(0).reorder(" << vars_str.str()
                    << ");\n";
        // clear the reorder variable list after using them (to avoid
        // accidentally trying to reorder) or if we consume these earlier (e.g.
        // when using rfactor)
        halide_reorder.clear();
        return C;
    }

    void add_to_reorder(const std::string var_name)
    {
        halide_reorder.push_back(var_name);
    }

    void schedule_fma(Halide::Func& C)
    {

        C.store_in(Halide::MemoryType::Register);
        for (auto const& e : order)
        {
            Halide::VarOrRVar current_var = get_current_dim_obj(e.first);
            add_to_reorder(current_var.name());

            if (e.second > 1)
            {
                split(C, e.first, e.second);
            }
        }
        Halide::Func vectorized = vectorize(C);
        Halide::Func unrolled   = unroll(vectorized);
        reorder(unrolled);
    }

    std::int64_t
    compute_tensor_size(const std::map<std::string, int>& strides) const
    {
        std::int64_t size = 1;
        for (auto const& s : sizes)
        {
            if (strides.count(s.first))
                size += (s.second - 1) * strides.at(s.first);
        }
        return size;
    }

    void set_tensor_sizes()
    {
        A_size = compute_tensor_size(A_strides);
        B_size = compute_tensor_size(B_strides);
        set_C_dim_sizes();
    }

private:
    void set_jit_target_features()
    {
        halide_target.set_feature(Halide::Target::Feature::NoAsserts, true);
        halide_target.set_feature(Halide::Target::Feature::NoBoundsQuery, true);

        // assertions to make sure picked up correctly
        assert(halide_target.has_feature(Halide::Target::Feature::NoAsserts));
        assert(
            halide_target.has_feature(Halide::Target::Feature::NoBoundsQuery));

        if (std::is_same_v<ISA, avx512>)
        {
            halide_target.set_feature(Halide::Target::Feature::AVX512, true);
            assert(halide_target.has_feature(Halide::Target::Feature::AVX512));
        }
        else if (std::is_same_v<ISA, avx2>)
        {
            halide_target.set_feature(Halide::Target::Feature::AVX2, true);
            halide_target.set_feature(Halide::Target::Feature::AVX512, false);

            assert(halide_target.has_feature(Halide::Target::Feature::AVX2));
            assert(!halide_target.has_feature(Halide::Target::Feature::AVX512));
        }
        else
        {
            assert("Unhandled ISA" && false);
        }
    }

    void generate_halide_program()
    {
        set_tensor_sizes();
        set_reduction_dims();
        create_rdom();
        create_initial_vars_and_rvars();
        A_param        = create_buffer("A", A_size);
        B_param        = create_buffer("B", B_size);
        Halide::Func C = define_fma();
        bound_initial_vars(C);
        schedule_fma(C);
        halide_program = C;
    }

private:
    static constexpr int default_max_fmas_unrolled = 320;

public:
    LoopNestToHalide(
        std::vector<std::pair<std::string, int>> const& _order,
        std::map<std::string, int> const&               sizes,
        std::set<std::string> const&                    C_formula,
        std::set<std::string> const&                    A_formula,
        std::set<std::string> const&                    B_formula,
        std::map<std::string, int> const&               C_strides,
        std::map<std::string, int> const&               A_strides,
        std::map<std::string, int> const&               B_strides,
        std::optional<int> user_fma_unroll_limit           = std::nullopt,
        std::shared_ptr<halide_elementwise_op> elementwise = nullptr)
        : order(_order)
        , sizes(sizes)
        , elementwise(elementwise)
        , C_formula(C_formula)
        , A_formula(A_formula)
        , B_formula(B_formula)
        , C_strides(C_strides)
        , A_strides(A_strides)
        , B_strides(B_strides)
        , max_fmas_unrolled(user_fma_unroll_limit ? *user_fma_unroll_limit
                                                  : default_max_fmas_unrolled)

    {
        generate_halide_program();
        set_jit_target_features();
        halide_program.print_loop_nest();
        print_halide();
    }

public:
    void compile_jit()
    {
        // factoring out so we can time compilation time separately
        halide_program.compile_jit();
    }

    void compile_to_assembly(std::string const& filename,
                             std::string const& fn_name)
    {
        Halide::Buffer<float> A_buf(nullptr, A_size);
        Halide::Buffer<float> B_buf(nullptr, B_size);
        Halide::Buffer<float> C_buf(nullptr, C_dim_sizes);

        Halide::Argument A_arg(A_buf);
        Halide::Argument B_arg(B_buf);
        Halide::Argument C_arg(C_buf);

        halide_program.compile_to_assembly(filename, {A_arg, B_arg}, fn_name);
    }

    void run_on_aligned_data(float* C_mem, float* A_mem, float* B_mem)
    {
        // Note: this function can throw a segfault if
        // run on data that is 1) not aligned to the appropriate
        // address multiple (32 for AVX2, 64 for AVX512), or 2)
        // the pointers are not a multiple of a vector size
        // (8 for AVX2 and 16 for AVX512)
        // we still need to put the pointers into appropriate
        // Halide buffers before we can .realize

        Halide::Buffer<float> C_buf(C_mem, C_dim_sizes);

        halide_buffer_t* A_param_buf = A_param.get()->raw_buffer();
        A_param_buf->host            = (uint8_t*)A_mem;

        halide_buffer_t* B_param_buf = B_param.get()->raw_buffer();
        B_param_buf->host            = (uint8_t*)B_mem;

        halide_program.realize(C_buf);
    }

    void run_on_unaligned_data(float* C_mem, float* A_mem, float* B_mem)
    {
        // Note: not really intended for use in benchmarking, but included
        // for completeness, in case a user wants to run on arbitrarily
        // allocated input data
        //
        // we allocate our own pointers with necessary alignment and vector
        // size multiples, and memcpy over the inputs before running as
        // aligned then return back to original memory
        alignas(alignment) float A_aligned[int(
            std::ceil(A_size / float(vector_size)) * vector_size)];
        alignas(alignment) float B_aligned[int(
            std::ceil(B_size / float(vector_size)) * vector_size)];
        alignas(alignment) float C_aligned[int(
            std::ceil(C_size / float(vector_size)) * vector_size)];

        memcpy(A_mem, A_aligned, A_size * 4);
        memcpy(B_mem, B_aligned, B_size * 4);
        memcpy(C_mem, C_aligned, C_size * 4);

        run_on_aligned_data(C_aligned, A_aligned, B_aligned);

        memcpy(C_aligned, C_mem, C_size * 4);
    }

    void dump_program(const std::string& file_name)
    {
        halide_program.compile_to_static_library(file_name, {A_param, B_param},
                                                 file_name + "_run");
        halide_program.compile_to_lowered_stmt(file_name + ".html",
                                               {A_param, B_param},
                                               Halide::StmtOutputFormat::HTML);
    }

    void print_halide()
    {
        std::cout << "// Halide code\n" << halide_code.str() << std::endl;
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
