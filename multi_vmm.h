#pragma once

#include <cassert>

namespace facebook
{
namespace sysml
{
namespace aot
{

// The main usage of the multi_vmm class is to increase the amount of
// independent operations when accumulating to a single vector
// register.  This is accomplished by using multiple vector registers
// which are reduced to a single one at the end.  Each of the size_
// registers is independent of all the other ones.

template <class Vmm>
class multi_vmm
{
private:
    int size_    = 0;
    int first_   = 0;
    int current_ = 0;

public:
    multi_vmm() {}

    multi_vmm(int s, int f)
        : size_(s)
        , first_(f)
        , current_(0)
    {
        assert(s > 0);
    }

    multi_vmm(multi_vmm const&) = delete;
    multi_vmm& operator=(multi_vmm const&) = delete;

    multi_vmm(multi_vmm&& o) { *this = std::move(o); }

    multi_vmm& operator=(multi_vmm&& o)
    {
        assert(o.size_ > 0);
        size_    = o.size_;
        first_   = o.first_;
        current_ = o.current_;
        return *this;
    }

    int size() const { return size_; }

    Vmm operator++(int)
    {
        int c    = current_;
        current_ = (current_ + 1) % size_;
        return Vmm(first_ + c);
    }

    Vmm operator[](int s) const
    {
        assert(s < size_);
        return Vmm(first_ + s);
    }

    Vmm operator++()
    {
        current_ = (current_ + 1) % size_;
        return Vmm(first_ + current_);
    }

    Vmm current() const { return Vmm(first_ + current_); }

    Vmm first() const { return Vmm(first_); }

    template <class Jitter>
    void half(Jitter& jitter)
    {
        int h = (size_ + 1) / 2;
        for (int i = 0; i + h < size_; ++i)
        {
            jitter.vaddps(Vmm(first_ + i), Vmm(first_ + i),
                          Vmm(first_ + i + h));
        }
        size_    = h;
        current_ = 0;
    }

    template <class Jitter>
    void reduce(Jitter& jitter)
    {
        while (size_ > 1)
        {
            half(jitter);
        }
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
