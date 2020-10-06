// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "dabun/common.hpp"

namespace dabun
{
namespace loop_tree
{

struct program_node_info
{
    std::int64_t const flops           = 0;
    std::int64_t const effective_flops = 0;

    std::string to_string() const
    {
        return std::string("program - FLOPs: ") + std::to_string(flops) +
               ", effective FLOPs: " + std::to_string(effective_flops);
    }
};

struct compute_node_info
{
    std::int64_t const flops           = 0;
    std::int64_t const effective_flops = 0;

    std::string to_string() const { return "compute node: 2 FLOPs"; }
};

struct compiled_loop_nest_node_info
{
    std::int64_t const flops           = 0;
    std::int64_t const effective_flops = 0;
    std::string const  asm_dump        = "";

    access_kind A_access_kind;
    access_kind B_access_kind;
    access_kind C_access_kind;

    std::string to_string() const
    {
        return std::string("compiled loop_nest - FLOPs: ") +
               std::to_string(flops) +
               ", effective FLOPs: " + std::to_string(effective_flops) +
               ", A access: " + dabun::to_string(A_access_kind) +
               ", B access: " + dabun::to_string(B_access_kind) +
               ", C access: " + dabun::to_string(C_access_kind);
    }
};

struct transpose_node_info
{
    std::int64_t const flops           = 0;
    std::int64_t const effective_flops = 0;

    std::string to_string() const { return "transpose_node"; }
};

struct compiled_transpose_node_info
{
    std::int64_t const flops           = 0;
    std::int64_t const effective_flops = 0;
    std::string const  asm_dump        = "";

    std::string to_string() const { return "compiled_transpose_node"; }
};

struct for_loop_node_info
{
    std::int64_t const flops           = 0;
    std::int64_t const effective_flops = 0;

    std::string const  var_name = "";
    std::int64_t const steps    = 0;
    std::int64_t const delta    = 0;

    std::string to_string() const
    {
        return std::string("for_loop - FLOPs: ") + std::to_string(flops) +
               ", effective FLOPs: " + std::to_string(effective_flops) +
               ", var" + var_name + ", steps: " + std::to_string(steps) +
               ", delta: " + std::to_string(delta);
    }
};

using node_info =
    std::variant<compute_node_info, compiled_loop_nest_node_info,
                 transpose_node_info, compiled_transpose_node_info,
                 for_loop_node_info, program_node_info>;

struct node_report;

using report_vector = std::vector<std::shared_ptr<node_report>>;

struct node_report
{
    node_info     info;
    report_vector children;

    node_report(node_info i)
        : info(i)
    {
    }

    node_report(node_info i, report_vector&& c)
        : info(i)
        , children(std::move(c))
    {
    }
};

inline void print_report_helper(std::ostringstream&  oss,
                                report_vector const& report, int indent = 0)
{
    for (auto const& r : report)
    {
        std::visit(
            [&](auto const& i) {
                oss << std::string(indent, '|') << i.to_string() << '\n';
            },
            r->info);
        print_report_helper(oss, r->children, indent + 2);
    }
}

inline std::string print_report(report_vector const& report, int indent)
{
    std::ostringstream oss;
    print_report_helper(oss, report, indent);
    return oss.str();
}

inline std::string print_report(std::shared_ptr<node_report> const& node,
                                int                                 indent = 0)
{
    std::ostringstream oss;
    std::visit(
        [&](auto const& i) {
            oss << std::string(indent, '|') << i.to_string() << '\n';
        },
        node->info);
    print_report_helper(oss, node->children, indent + 2);
    return oss.str();
}

} // namespace loop_tree
} // namespace dabun
