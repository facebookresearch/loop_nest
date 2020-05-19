#pragma once

#include <cassert>
#include <cstddef>
#include <map>
#include <set>

namespace facebook
{
namespace sysml
{
namespace aot
{

// A queue that holds values of type T and a count for each value.  We
// can queue the most abundant value, remove it, as well as increase
// or decrease instances of each value by 1.

template <class T>
struct most_frequent_queue
{
private:
    std::map<std::size_t, std::set<T>> sorted_;
    std::map<T, std::size_t>           counts_;

    std::size_t remove_existing(T const& v)
    {
        assert(counts_.count(v) > 0);

        std::size_t s = counts_[v];

        assert(sorted_.count(s) > 0);

        auto& bucket = sorted_[s];

        assert(bucket.count(v) > 0);

        bucket.erase(v);
        if (bucket.size() == 0)
        {
            sorted_.erase(s);
        }

        counts_.erase(v);
        return s;
    }

    void add_with_count(T const& v, std::size_t s)
    {
        counts_[v] = s;
        assert(sorted_.count(s) == 0 || sorted_[s].count(v) == 0);
        sorted_[s].insert(v);
    }

public:
    std::size_t size() const { return counts_.size(); }

    T top() const
    {
        assert(size() > 0);
        return *(sorted_.crbegin()->second.cbegin());
    }

    T get_top_then_pop()
    {
        T ret = top();
        pop();
        return ret;
    }

    void pop()
    {
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

    void inc(T const& v)
    {
        if (counts_.count(v))
        {
            auto s = remove_existing(v);
            add_with_count(v, s + 1);
        }
        else
        {
            add_with_count(v, 1);
        }
    }

    void dec(T const& v)
    {
        assert(counts_.count(v));
        auto s = remove_existing(v);
        if (--s > 0)
        {
            add_with_count(v, s);
        }
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
