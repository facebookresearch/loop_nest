// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/core.hpp"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dabun
{

// TODO(zi) include the rest when they get an ISA independent implementation

class LN_arguments
{

private:
    std::vector<std::pair<std::string, int>> order_;
    std::map<std::string, int>               sizes_;
    std::set<std::string>                    C_axes_;
    std::set<std::string>                    A_axes_;
    std::set<std::string>                    B_axes_;
    std::map<std::string, int>               C_strides_;
    std::map<std::string, int>               A_strides_;
    std::map<std::string, int>               B_strides_;

public:
    std::vector<std::pair<std::string, int>> const& get_order() const
    {
        return order_;
    }

    std::map<std::string, int> const& get_sizes() const { return sizes_; }

    std::set<std::string> const& get_C_axes() const { return C_axes_; }

    std::set<std::string> const& get_A_axes() const { return A_axes_; }

    std::set<std::string> const& get_B_axes() const { return B_axes_; }

    std::map<std::string, int> const& get_C_strides() const
    {
        return C_strides_;
    }

    std::map<std::string, int> const& get_A_strides() const
    {
        return A_strides_;
    }

    std::map<std::string, int> const& get_B_strides() const
    {
        return B_strides_;
    }

public:
    LN_arguments& append_loop(std::string const& d, int s)
    {
        order_.push_back({d, s});
        return *this;
    }

    LN_arguments& append_loop(std::pair<std::string, int> const& l)
    {
        order_.push_back(l);
        return *this;
    }

    LN_arguments&
    append_loops(std::vector<std::pair<std::string, int>> const& ds)
    {
        order_.insert(order_.end(), ds.begin(), ds.end());
        return *this;
    }

    LN_arguments& size(std::string const& d, int s)
    {
        strong_assert(sizes_.count(d) == 0);
        sizes_[d] = s;
        return *this;
    }

    LN_arguments& size(std::pair<std::string, int> const& s)
    {
        size(s.first, s.second);
        return *this;
    }

    LN_arguments& sizes(std::vector<std::pair<std::string, int>> const& ds)
    {
        for (auto const& s : ds)
        {
            size(s);
        }
        return *this;
    }

    LN_arguments& C_axis(std::string const& d)
    {
        strong_assert(C_axes_.count(d) == 0);
        C_axes_.insert(d);
        return *this;
    }

    LN_arguments& C_axes(std::vector<std::string> const& ds)
    {
        for (auto const& x : ds)
        {
            C_axis(x);
        }
        return *this;
    }

    LN_arguments& A_axis(std::string const& d)
    {
        strong_assert(A_axes_.count(d) == 0);
        A_axes_.insert(d);
        return *this;
    }

    LN_arguments& A_axes(std::vector<std::string> const& ds)
    {
        for (auto const& x : ds)
        {
            A_axis(x);
        }
        return *this;
    }

    LN_arguments& B_axis(std::string const& d)
    {
        strong_assert(B_axes_.count(d) == 0);
        B_axes_.insert(d);
        return *this;
    }

    LN_arguments& B_axes(std::vector<std::string> const& ds)
    {
        for (auto const& x : ds)
        {
            B_axis(x);
        }
        return *this;
    }

    LN_arguments& C_stride(std::string const& d, int s)
    {
        strong_assert(C_strides_.count(d) == 0);
        C_strides_[d] = s;
        return *this;
    }

    LN_arguments& C_stride(std::pair<std::string, int> const s)
    {
        C_stride(s.first, s.second);
        return *this;
    }

    LN_arguments& C_strides(std::vector<std::pair<std::string, int>> const& ds)
    {
        for (auto const& x : ds)
        {
            C_stride(x);
        }
        return *this;
    }

    LN_arguments& A_stride(std::pair<std::string, int> const s)
    {
        A_stride(s.first, s.second);
        return *this;
    }

    LN_arguments& A_stride(std::string const& d, int s)
    {
        strong_assert(A_strides_.count(d) == 0);
        A_strides_[d] = s;
        return *this;
    }

    LN_arguments& A_strides(std::vector<std::pair<std::string, int>> const& ds)
    {
        for (auto const& x : ds)
        {
            A_stride(x);
        }
        return *this;
    }

    LN_arguments& B_stride(std::string const& d, int s)
    {
        strong_assert(B_strides_.count(d) == 0);
        B_strides_[d] = s;
        return *this;
    }

    LN_arguments& B_stride(std::pair<std::string, int> const s)
    {
        B_stride(s.first, s.second);
        return *this;
    }

    LN_arguments& B_strides(std::vector<std::pair<std::string, int>> const& ds)
    {
        for (auto const& x : ds)
        {
            B_stride(x);
        }
        return *this;
    }
};

inline LN_arguments LN_loop(std::string const& d, int s)
{
    LN_arguments ret;
    ret.append_loop(d, s);
    return ret;
}

inline LN_arguments LN_loops(std::vector<std::pair<std::string, int>> const& ds)
{
    LN_arguments ret;
    ret.append_loops(ds);
    return ret;
}

inline LN_arguments LN_size(std::string const& d, int s)
{
    LN_arguments ret;
    ret.size(d, s);
    return ret;
}

inline LN_arguments LN_sizes(std::vector<std::pair<std::string, int>> const& ds)
{
    LN_arguments ret;
    ret.sizes(ds);
    return ret;
}

inline LN_arguments LN_C_axis(std::string const& d)
{
    LN_arguments ret;
    ret.C_axis(d);
    return ret;
}

inline LN_arguments C_axes(std::vector<std::string> const& ds)
{
    LN_arguments ret;
    ret.C_axes(ds);
    return ret;
}

inline LN_arguments A_axis(std::string const& d)
{
    LN_arguments ret;
    ret.A_axis(d);
    return ret;
}

inline LN_arguments A_axes(std::vector<std::string> const& ds)
{
    LN_arguments ret;
    ret.A_axes(ds);
    return ret;
}

inline LN_arguments B_axis(std::string const& d)
{
    LN_arguments ret;
    ret.B_axis(d);
    return ret;
}

inline LN_arguments B_axes(std::vector<std::string> const& ds)
{
    LN_arguments ret;
    ret.B_axes(ds);
    return ret;
}

inline LN_arguments C_stride(std::string const& d, int s)
{
    LN_arguments ret;
    ret.C_stride(d, s);
    return ret;
}

inline LN_arguments
C_strides(std::vector<std::pair<std::string, int>> const& ds)
{
    LN_arguments ret;
    ret.C_strides(ds);
    return ret;
}

inline LN_arguments A_stride(std::string const& d, int s)
{
    LN_arguments ret;
    ret.A_stride(d, s);
    return ret;
}

inline LN_arguments
A_strides(std::vector<std::pair<std::string, int>> const& ds)
{
    LN_arguments ret;
    ret.A_strides(ds);
    return ret;
}

inline LN_arguments B_stride(std::string const& d, int s)
{
    LN_arguments ret;
    ret.B_stride(d, s);
    return ret;
}

inline LN_arguments
B_strides(std::vector<std::pair<std::string, int>> const& ds)
{
    LN_arguments ret;
    ret.B_strides(ds);
    return ret;
}

} // namespace dabun
