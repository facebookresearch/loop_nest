#pragma once

#include <cassert>

namespace facebook
{
namespace sysml
{
namespace aot
{

// The main usage of the multi_vreg class is to increase the amount of
// independent operations when accumulating to a single vector
// register.  This is accomplished by using multiple vector registers
// which are reduced to a single one at the end.  Each of the size_
// registers is independent of all the other ones.

template <class VReg>
class multi_vreg
{
private:
    int size_    = 0;
    int first_   = 0;
    int current_ = 0;
    int vlen_    = 4;

public:
    multi_vreg() {}

    multi_vreg(int s, int f)
        : size_(s)
        , first_(f)
        , current_(0)
    {
        assert(s > 0);
    }

    multi_vreg(multi_vreg const&) = delete;
    multi_vreg& operator=(multi_vreg const&) = delete;

    multi_vreg(multi_vreg&& o) { *this = std::move(o); }

    multi_vreg& operator=(multi_vreg&& o)
    {
        assert(o.size_ > 0);
        size_    = o.size_;
        first_   = o.first_;
        current_ = o.current_;
        return *this;
    }

    int size() const { return size_; }

    VReg operator++(int)
    {
        int c    = current_;
        current_ = (current_ + 1) % size_;
        return VReg(first_ + c);
    }

    VReg operator[](int s) const
    {
        assert(s < size_);
        return VReg(first_ + s);
    }

    VReg operator++()
    {
        current_ = (current_ + 1) % size_;
        return VReg(first_ + current_);
    }

    VReg current() const { return VReg(first_ + current_); }

    VReg first() const { return VReg(first_); }

    template <class Jitter>
    void half(Jitter& jitter)
    {
        int h = (size_ + 1) / 2;
        for (int i = 0; i + h < size_; ++i)
        {
            jitter.fadd(VReg(first_ + i).s4, VReg(first_ + i).s4,
                        VReg(first_ + i + h).s4);
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

    template <class Jitter>
    void full_reduce(Jitter& jitter, int mask = 4)
    {
        reduce(jitter);
        assert(size_ == 1);
        if (mask == 3)
        {
            jitter.ins(VReg(first_).s4[3], jitter.w4);
        }
        if (mask > 2)
        {
            jitter.faddp(VReg(first_).s4, VReg(first_).s4, VReg(first_).s4);
        }
        if (mask > 1)
        {
            jitter.faddp(VReg(first_).s2, VReg(first_).s2, VReg(first_).s2);
        }
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
