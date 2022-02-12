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

class loop_nest_descriptor;

class loop_nest_verified_descriptor final
{
private:
    loop_nest_descriptor const *const desc_;

    friend class loop_nest_descriptor;

    loop_nest_verified_descriptor(loop_nest_descriptor const *desc)
        : desc_(desc)
    {
    }

public:
    std::vector<std::pair<std::string, int>> const &get_order() const;
    std::map<std::string, int> const               &get_sizes() const;
    std::set<std::string> const                    &get_C_axes() const;
    std::set<std::string> const                    &get_A_axes() const;
    std::set<std::string> const                    &get_B_axes() const;
    std::map<std::string, int> const               &get_C_strides() const;
    std::map<std::string, int> const               &get_A_strides() const;
    std::map<std::string, int> const               &get_B_strides() const;
};

class loop_nest_descriptor
{
public:
    struct required_sizes
    {
        std::int64_t C_size = 0;
        std::int64_t A_size = 0;
        std::int64_t B_size = 0;
    };

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
    operator loop_nest_verified_descriptor() const
    {
        // TODO(zi) Should do a full representation check here
        return loop_nest_verified_descriptor(this);
    }

    bool is_reduction_extent(std::string const &extent) const noexcept
    {
        return A_axes_.count(extent) && C_axes_.count(extent) == 0;
    }

    std::int64_t reduction_size() const noexcept
    {
        std::int64_t ret = 1;

        for (auto const &s : sizes_)
        {
            if (is_reduction_extent(s.first))
            {
                ret *= s.second;
            }
        }

        return ret;
    }

    required_sizes get_required_sizes() const
    {
        required_sizes ret{1, 1, 1};

        for (auto const &s : sizes_)
        {
            if (C_strides_.count(s.first))
                ret.C_size += (s.second - 1) * C_strides_.at(s.first);
            if (A_strides_.count(s.first))
                ret.A_size += (s.second - 1) * A_strides_.at(s.first);
            if (B_strides_.count(s.first))
                ret.B_size += (s.second - 1) * B_strides_.at(s.first);
        }

        return ret;
    }

    std::vector<std::pair<std::string, int>> const &get_order() const
    {
        return order_;
    }

    std::map<std::string, int> const &get_sizes() const { return sizes_; }

    std::set<std::string> const &get_C_axes() const { return C_axes_; }

    std::set<std::string> const &get_A_axes() const { return A_axes_; }

    std::set<std::string> const &get_B_axes() const { return B_axes_; }

    std::map<std::string, int> const &get_C_strides() const
    {
        return C_strides_;
    }

    std::map<std::string, int> const &get_A_strides() const
    {
        return A_strides_;
    }

    std::map<std::string, int> const &get_B_strides() const
    {
        return B_strides_;
    }

public:
    loop_nest_descriptor &append_loop(std::string const &d, int s)
    {
        order_.push_back({d, s});
        return *this;
    }

    loop_nest_descriptor &append_loop(std::pair<std::string, int> const &l)
    {
        order_.push_back(l);
        return *this;
    }

    loop_nest_descriptor &
    append_loops(std::vector<std::pair<std::string, int>> const &ds)
    {
        order_.insert(order_.end(), ds.begin(), ds.end());
        return *this;
    }

    loop_nest_descriptor &size(std::string const &d, int s)
    {
        strong_assert(sizes_.count(d) == 0);
        sizes_[d] = s;
        return *this;
    }

    loop_nest_descriptor &size(std::pair<std::string, int> const &s)
    {
        size(s.first, s.second);
        return *this;
    }

    loop_nest_descriptor &
    sizes(std::vector<std::pair<std::string, int>> const &ds)
    {
        for (auto const &s : ds)
        {
            size(s);
        }
        return *this;
    }

    loop_nest_descriptor &C_axis(std::string const &d)
    {
        strong_assert(C_axes_.count(d) == 0);
        C_axes_.insert(d);
        return *this;
    }

    loop_nest_descriptor &C_axes(std::vector<std::string> const &ds)
    {
        for (auto const &x : ds)
        {
            C_axis(x);
        }
        return *this;
    }

    loop_nest_descriptor &A_axis(std::string const &d)
    {
        strong_assert(A_axes_.count(d) == 0);
        A_axes_.insert(d);
        return *this;
    }

    loop_nest_descriptor &A_axes(std::vector<std::string> const &ds)
    {
        for (auto const &x : ds)
        {
            A_axis(x);
        }
        return *this;
    }

    loop_nest_descriptor &B_axis(std::string const &d)
    {
        strong_assert(B_axes_.count(d) == 0);
        B_axes_.insert(d);
        return *this;
    }

    loop_nest_descriptor &B_axes(std::vector<std::string> const &ds)
    {
        for (auto const &x : ds)
        {
            B_axis(x);
        }
        return *this;
    }

    loop_nest_descriptor &C_stride(std::string const &d, int s)
    {
        strong_assert(C_strides_.count(d) == 0);
        C_strides_[d] = s;
        return *this;
    }

    loop_nest_descriptor &C_stride(std::pair<std::string, int> const s)
    {
        C_stride(s.first, s.second);
        return *this;
    }

    loop_nest_descriptor &
    C_strides(std::vector<std::pair<std::string, int>> const &ds)
    {
        for (auto const &x : ds)
        {
            C_stride(x);
        }
        return *this;
    }

    loop_nest_descriptor &A_stride(std::pair<std::string, int> const s)
    {
        A_stride(s.first, s.second);
        return *this;
    }

    loop_nest_descriptor &A_stride(std::string const &d, int s)
    {
        strong_assert(A_strides_.count(d) == 0);
        A_strides_[d] = s;
        return *this;
    }

    loop_nest_descriptor &
    A_strides(std::vector<std::pair<std::string, int>> const &ds)
    {
        for (auto const &x : ds)
        {
            A_stride(x);
        }
        return *this;
    }

    loop_nest_descriptor &B_stride(std::string const &d, int s)
    {
        strong_assert(B_strides_.count(d) == 0);
        B_strides_[d] = s;
        return *this;
    }

    loop_nest_descriptor &B_stride(std::pair<std::string, int> const s)
    {
        B_stride(s.first, s.second);
        return *this;
    }

    loop_nest_descriptor &
    B_strides(std::vector<std::pair<std::string, int>> const &ds)
    {
        for (auto const &x : ds)
        {
            B_stride(x);
        }
        return *this;
    }

public:
    loop_nest_descriptor() {}

    loop_nest_descriptor(loop_nest_descriptor const &) = default;
    loop_nest_descriptor(loop_nest_descriptor &&)      = default;

    loop_nest_descriptor &operator=(loop_nest_descriptor const &) = default;
    loop_nest_descriptor &operator=(loop_nest_descriptor &&) = default;

    loop_nest_descriptor(std::vector<std::pair<std::string, int>> const &order,
                         std::map<std::string, int> const               &sizes,
                         std::set<std::string> const                    &C_axes,
                         std::set<std::string> const                    &A_axes,
                         std::set<std::string> const                    &B_axes,
                         std::map<std::string, int> const &C_strides,
                         std::map<std::string, int> const &A_strides,
                         std::map<std::string, int> const &B_strides)
        : order_(order)
        , sizes_(sizes)
        , C_axes_(C_axes)
        , A_axes_(A_axes)
        , B_axes_(B_axes)
        , C_strides_(C_strides)
        , A_strides_(A_strides)
        , B_strides_(B_strides)
    {
    }
};

inline loop_nest_descriptor LN_loop(std::string const &d, int s)
{
    loop_nest_descriptor ret;
    ret.append_loop(d, s);
    return ret;
}

inline loop_nest_descriptor
LN_loops(std::vector<std::pair<std::string, int>> const &ds)
{
    loop_nest_descriptor ret;
    ret.append_loops(ds);
    return ret;
}

inline loop_nest_descriptor LN_size(std::string const &d, int s)
{
    loop_nest_descriptor ret;
    ret.size(d, s);
    return ret;
}

inline loop_nest_descriptor
LN_sizes(std::vector<std::pair<std::string, int>> const &ds)
{
    loop_nest_descriptor ret;
    ret.sizes(ds);
    return ret;
}

inline loop_nest_descriptor LN_C_axis(std::string const &d)
{
    loop_nest_descriptor ret;
    ret.C_axis(d);
    return ret;
}

inline loop_nest_descriptor C_axes(std::vector<std::string> const &ds)
{
    loop_nest_descriptor ret;
    ret.C_axes(ds);
    return ret;
}

inline loop_nest_descriptor A_axis(std::string const &d)
{
    loop_nest_descriptor ret;
    ret.A_axis(d);
    return ret;
}

inline loop_nest_descriptor A_axes(std::vector<std::string> const &ds)
{
    loop_nest_descriptor ret;
    ret.A_axes(ds);
    return ret;
}

inline loop_nest_descriptor B_axis(std::string const &d)
{
    loop_nest_descriptor ret;
    ret.B_axis(d);
    return ret;
}

inline loop_nest_descriptor B_axes(std::vector<std::string> const &ds)
{
    loop_nest_descriptor ret;
    ret.B_axes(ds);
    return ret;
}

inline loop_nest_descriptor C_stride(std::string const &d, int s)
{
    loop_nest_descriptor ret;
    ret.C_stride(d, s);
    return ret;
}

inline loop_nest_descriptor
C_strides(std::vector<std::pair<std::string, int>> const &ds)
{
    loop_nest_descriptor ret;
    ret.C_strides(ds);
    return ret;
}

inline loop_nest_descriptor A_stride(std::string const &d, int s)
{
    loop_nest_descriptor ret;
    ret.A_stride(d, s);
    return ret;
}

inline loop_nest_descriptor
A_strides(std::vector<std::pair<std::string, int>> const &ds)
{
    loop_nest_descriptor ret;
    ret.A_strides(ds);
    return ret;
}

inline loop_nest_descriptor B_stride(std::string const &d, int s)
{
    loop_nest_descriptor ret;
    ret.B_stride(d, s);
    return ret;
}

inline loop_nest_descriptor
B_strides(std::vector<std::pair<std::string, int>> const &ds)
{
    loop_nest_descriptor ret;
    ret.B_strides(ds);
    return ret;
}

inline std::vector<std::pair<std::string, int>> const &
loop_nest_verified_descriptor::get_order() const
{
    return desc_->get_order();
}

inline std::map<std::string, int> const &
loop_nest_verified_descriptor::get_sizes() const
{
    return desc_->get_sizes();
}

inline std::set<std::string> const &
loop_nest_verified_descriptor::get_C_axes() const
{
    return desc_->get_C_axes();
}

inline std::set<std::string> const &
loop_nest_verified_descriptor::get_A_axes() const
{
    return desc_->get_A_axes();
}

inline std::set<std::string> const &
loop_nest_verified_descriptor::get_B_axes() const
{
    return desc_->get_B_axes();
}

inline std::map<std::string, int> const &
loop_nest_verified_descriptor::get_C_strides() const
{
    return desc_->get_C_strides();
}

inline std::map<std::string, int> const &
loop_nest_verified_descriptor::get_A_strides() const
{
    return desc_->get_A_strides();
}

inline std::map<std::string, int> const &
loop_nest_verified_descriptor::get_B_strides() const
{
    return desc_->get_B_strides();
}

} // namespace dabun
