// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/numeric.hpp"

#include <cassert>
#include <type_traits>

namespace dabun
{
namespace tensillica
{

// The main usage of the multi_vmm class is to increase the amount of
// independent operations when accumulating to a single vector
// register.  This is accomplished by using multiple vector registers
// which are reduced to a single one at the end.  Each of the size_
// registers is independent of all the other ones.

template <class VReg, class SReg, class Reg32>
class multi_vmm
{
private:
    int size_          = 0;
    int first_         = 0;
    int current_       = 0;
    int vlen_          = 4;
    int original_size_ = 0;

public:
    multi_vmm() {}

    multi_vmm(int s, int f)
        : size_(s)
        , first_(f)
        , current_(0)
        , original_size_(s)
    {
        assert(s > 0);
    }

    void reset()
    {
        size_    = original_size_;
        current_ = 0;
    }

    multi_vmm(multi_vmm const&)            = delete;
    multi_vmm& operator=(multi_vmm const&) = delete;

    multi_vmm(multi_vmm&& o) { *this = std::move(o); }

    multi_vmm& operator=(multi_vmm&& o)
    {
        assert(o.size_ > 0);
        size_          = o.size_;
        first_         = o.first_;
        current_       = o.current_;
        original_size_ = o.original_size_;
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

    template <class Code_Generator>
    void half(Code_Generator& code_generator)
    {
        int h = (size_ + 1) / 2;
        for (int i = 0; i + h < size_; ++i)
        {
            code_generator.fadd(VReg(first_ + i).s4, VReg(first_ + i).s4,
                                VReg(first_ + i + h).s4);
        }
        size_    = h;
        current_ = 0;
    }

    template <class Code_Generator>
    void reduce(Code_Generator& code_generator)
    {
        while (size_ > 1)
        {
            half(code_generator);
        }
    }

    template <class Code_Generator>
    void full_reduce(Code_Generator& code_generator, int mask = 4,
                     int zero_vector = 0)
    {
        reduce(code_generator);
        assert(size_ == 1);

        {
            if (mask == 3)
            {
                // x4/w4 is zero reg by convention in the loop_nest.hpp
                code_generator.ins(VReg(first_).s4[3], Reg32(zero_vector));
            }
            if (mask > 2)
            {
                code_generator.faddp(VReg(first_).s4, VReg(first_).s4,
                                     VReg(first_).s4);
            }
            if (mask > 1)
            {
                code_generator.faddp(SReg(first_), VReg(first_).s2);
            }
        }
    }
};

} // namespace tensillica
} // namespace dabun
