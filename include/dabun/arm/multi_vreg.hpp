#pragma once

#include <cassert>
#include <type_traits>

namespace dabun
{
namespace arm
{

// The main usage of the multi_vreg class is to increase the amount of
// independent operations when accumulating to a single vector
// register.  This is accomplished by using multiple vector registers
// which are reduced to a single one at the end.  Each of the size_
// registers is independent of all the other ones.

template <class VReg, class SReg, class HReg>
class multi_vreg
{
private:
    int size_          = 0;
    int first_         = 0;
    int current_       = 0;
    int vlen_          = 4;
    int original_size_ = 0;

public:
    multi_vreg() {}

    multi_vreg(int s, int f)
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

    multi_vreg(multi_vreg const&) = delete;
    multi_vreg& operator=(multi_vreg const&) = delete;

    multi_vreg(multi_vreg&& o) { *this = std::move(o); }

    multi_vreg& operator=(multi_vreg&& o)
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

    template <class Float, class Code_Generator>
    void half(Code_Generator& code_generator)
    {
        static_assert(std::is_same_v<Float, float> ||
                      std::is_same_v<Float, fp16>);

        int h = (size_ + 1) / 2;
        for (int i = 0; i + h < size_; ++i)
        {
            if constexpr (std::is_same_v<Float, float>)
            {
                code_generator.fadd(VReg(first_ + i).s4, VReg(first_ + i).s4,
                                    VReg(first_ + i + h).s4);
            }
            else if constexpr (std::is_same_v<Float, fp16>)
            {
                code_generator.fadd(VReg(first_ + i).h8, VReg(first_ + i).h8,
                                    VReg(first_ + i + h).h8);
            }
        }
        size_    = h;
        current_ = 0;
    }

    template <class Float, class Code_Generator>
    void reduce(Code_Generator& code_generator)
    {
        static_assert(std::is_same_v<Float, float> ||
                      std::is_same_v<Float, fp16>);

        while (size_ > 1)
        {
            half<Float>(code_generator);
        }
    }

    template <class Float, class Code_Generator>
    void full_reduce(Code_Generator& code_generator, int mask = 4,
                     int zero_vector = 0)
    {
        static_assert(std::is_same_v<Float, float> ||
                      std::is_same_v<Float, fp16>);

        reduce<Float>(code_generator);
        assert(size_ == 1);

        if constexpr (std::is_same_v<Float, float>)
        {
            if (mask == 3)
            {
                // x4/w4 is zero reg by convention in the loop_nest.hpp
                code_generator.ins(VReg(first_).s4[3], code_generator.w4);
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
        else if constexpr (std::is_same_v<Float, fp16>)
        {
            switch (mask)
            {
            case 3:
                code_generator.ins(VReg(first_).h8[3], VReg(zero_vector).h8[3]);
                break;
            case 5:
                code_generator.ins(VReg(first_).h8[5], VReg(zero_vector).h8[5]);
                // fallthrough
            case 6:
                code_generator.ins(VReg(first_).s4[3], code_generator.w4);
                break;
            case 7:
                code_generator.ins(VReg(first_).h8[7], VReg(zero_vector).h8[7]);
                break;
            default:
                break;
            }

            if (mask > 4)
            {
                code_generator.faddp(VReg(first_).h8, VReg(first_).h8,
                                     VReg(first_).h8);
            }
            if (mask > 2)
            {
                code_generator.faddp(VReg(first_).h4, VReg(first_).h4,
                                     VReg(first_).h4);
            }
            if (mask > 1)
            {
                // TO DO HERE.
                code_generator.faddp(HReg(first_), VReg(first_).h2);
            }
        }
    }
};

} // namespace arm
} // namespace dabun
