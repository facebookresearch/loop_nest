// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace dabun
{
namespace loop_tree
{

struct program_node_info
{
    std::string to_string() const { return "program_node_info"; }
};

struct compute_node_info
{
    std::string to_string() const { return "compute node (2 FLOPs)"; }
};

struct compiled_loop_nest_node_info
{
    std::string to_string() const { return "compiled_loop_nest_node_info"; }
};

struct transpose_node_info
{
    std::string to_string() const { return "transpose_node_info"; }
};

struct compiled_transpose_node_info
{
    std::string to_string() const { return "compiled_transpose_node_info"; }
};

struct for_loop_node_info
{
    std::string to_string() const { return "for_loop_node_info"; }
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

inline std::string print_report(report_vector const& report, int indent = 0)
{
    std::ostringstream oss;
    print_report_helper(oss, report, indent);
    return oss.str();
}

} // namespace loop_tree
} // namespace dabun
