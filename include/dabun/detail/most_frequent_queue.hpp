// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cstddef>
#include <map>
#include <set>
#include <utility>

namespace dabun
{
namespace detail
{

// A queue that holds values of type T and a count for each value.  We
// can queue the most abundant value, remove it, as well as increase
// or decrease instances of each value by 1.

template <class T>
struct most_frequent_queue
{
private:
    using map_t = std::map<std::size_t, std::set<T>>;

    map_t                    sorted_;
    map_t                    ignored_;
    std::map<T, std::size_t> counts_;

    bool was_read_ = false;

    std::pair<map_t*, std::size_t> remove_existing(T const& v)
    {
        assert(counts_.count(v) > 0);

        std::size_t s = counts_[v];

        if (sorted_.count(s) > 0 && sorted_[s].count(v))
        {
            auto& bucket = sorted_[s];

            assert(bucket.count(v) > 0);

            bucket.erase(v);
            if (bucket.size() == 0)
            {
                sorted_.erase(s);
            }

            counts_.erase(v);
            return {&sorted_, s};
        }
        else
        {
            assert(ignored_.count(s) > 0);

            auto& bucket = ignored_[s];

            assert(bucket.count(v) > 0);

            bucket.erase(v);
            if (bucket.size() == 0)
            {
                ignored_.erase(s);
            }

            counts_.erase(v);
            return {&ignored_, s};
        }
    }

    void add_with_count(map_t* where, T const& v, std::size_t s)
    {
        counts_[v] = s;
        assert(where->count(s) == 0 || (*where)[s].count(v) == 0);
        (*where)[s].insert(v);
    }

    void pop_or_skip(map_t* /* where */) {}

public:
    std::size_t size() const { return sorted_.size(); }

    T top() const
    {
        assert(sorted_.size());
        return *(sorted_.crbegin()->second.cbegin());
    }

    T get_top_then_pop()
    {
        was_read_ = true;
        T ret     = top();
        pop();
        return ret;
    }

    void pop()
    {
        was_read_ = true;
        assert(sorted_.size() > 0);
        auto& slot = sorted_.rbegin()->second;

        assert(slot.size() > 0);
        auto const& v = *slot.begin();

        assert(counts_.count(v) > 0);
        counts_.erase(v);

        slot.erase(slot.begin());
        if (slot.size() == 0)
        {
            sorted_.erase(sorted_.rbegin()->first);
        }
    }

    void skip()
    {
        was_read_ = true;
        assert(sorted_.size() > 0);

        auto& slot = sorted_.rbegin()->second;

        assert(slot.size() > 0);
        auto const& v = *slot.begin();

        assert(counts_.count(v) > 0);

        slot.erase(slot.begin());
        if (slot.size() == 0)
        {
            sorted_.erase(sorted_.rbegin()->first);
        }

        ignored_[counts_[v]].insert(v);
    }

    void inc(T const& v)
    {
        assert(!was_read_);

        if (counts_.count(v))
        {
            auto s = remove_existing(v);
            add_with_count(s.first, v, s.second + 1);
        }
        else
        {
            add_with_count(&sorted_, v, 1);
        }
    }

    void dec(T const& v)
    {
        assert(was_read_);
        assert(counts_.count(v));

        auto s = remove_existing(v);
        if (--s.second > 0)
        {
            add_with_count(s.first, v, s.second);
        }
    }
};

} // namespace detail
} // namespace dabun
