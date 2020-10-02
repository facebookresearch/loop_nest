// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dabun/loop_tree/types.hpp"

namespace dabun
{
namespace loop_tree
{
namespace utility
{

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

} // namespace utility
} // namespace loop_tree
} // namespace dabun
