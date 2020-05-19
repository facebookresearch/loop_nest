#pragma once

/*

// The sizes were experimentally obtained.

AVX2 ptr[reg_base + off]

 |----------------------+-----------------------------------------------|
 | off                  | bytes                                         |
 |----------------------+-----------------------------------------------|
 | 0                    | 5  + 1  when reg_base in {rsp, rbp, r12, r13} |
 | [-0x80, 0x7c] by 0x4 | 6  + 1  when reg_base in {rsp, r12}           |
 | rest                 | 9  + 1  when reg_base in {rsp, r12}           |
 |----------------------+-----------------------------------------------|

 AVX2 ptr[reg_base + reg_index * [1|2|4|8] + off]

 |----------------------+-------------------------------------|
 | off                  |                               bytes |
 |----------------------+-------------------------------------|
 | 0                    | 6  + 1  when reg_base in {rbp, r13} |
 | [-0x80, 0x7c] by 0x4 |                                   7 |
 | rest                 |                                  10 |
 |----------------------+-------------------------------------|


 AVX512 ptr[reg_base + off]

 |---------------------------+-----------------------------------------------|
 | off                       | bytes                                         |
 |---------------------------+-----------------------------------------------|
 | 0                         | 6  + 1  when reg_base in {rsp, rbp, r12, r13} |
 | [-0x1fc0, 0x2000] by 0x40 | 7  + 1  when reg_base in {rsp, r12}           |
 | rest                      | 10  + 1  when reg_base in {rsp, r12}          |
 |---------------------------+-----------------------------------------------|

 AVX512 ptr_b[reg_base + off]

 |------------------------+-----------------------------------------------|
 | off                    | bytes                                         |
 |------------------------+-----------------------------------------------|
 | 0                      | 6  + 1  when reg_base in {rsp, rbp, r12, r13} |
 | [-0x1fc, 0x200] by 0x4 | 7  + 1  when reg_base in {rsp, r12}           |
 | rest                   | 10  + 1  when reg_base in {rsp, r12}          |
 |------------------------+-----------------------------------------------|


 AVX512 ptr[reg_base + reg_index * [1|2|4|8] + off]

 |---------------------------+-------------------------------------|
 | off                       |                               bytes |
 |---------------------------+-------------------------------------|
 | 0                         | 7  + 1  when reg_base in {rbp, r13} |
 | [-0x1fc0, 0x2000] by 0x40 |                                   8 |
 | rest                      |                                  11 |
 |---------------------------+-------------------------------------|


 AVX512 ptr_b[reg_base + reg_index * [1|2|4|8] + off]

 |------------------------+-------------------------------------|
 | off                    |                               bytes |
 |------------------------+-------------------------------------|
 | 0                      | 7  + 1  when reg_base in {rbp, r13} |
 | [-0x1fc, 0x200] by 0x4 |                                   8 |
 | rest                   |                                  11 |
 |------------------------+-------------------------------------|

*/

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"

#include <map>
#include <set>
#include <vector>

namespace facebook
{
namespace sysml
{

class address_packer
{
public:
    virtual ~address_packer(){};

    virtual void          initialize()                   = 0;
    virtual void          loop_prologue()                = 0;
    virtual void          move_to(int)                   = 0;
    virtual void          advance(int)                   = 0;
    virtual Xbyak::RegExp get_address(int)               = 0;
    virtual Xbyak::RegExp get_address_without_index(int) = 0;
    virtual void          restore()                      = 0;
};

class trivial_address_packer : public address_packer
{
private:
    Xbyak::CodeGenerator* code_generator_;
    Xbyak::Reg64          reg_;
    int                   offset_ = 0;

public:
    trivial_address_packer(Xbyak::CodeGenerator* code_generator,
                           Xbyak::Reg64          reg)
        : code_generator_(code_generator)
        , reg_(reg)
    {
    }

    void loop_prologue() override {}

    void initialize() override {}

    void move_to(int location) override
    {
        if (offset_ != location)
        {
            code_generator_->add(reg_, (location - offset_));
            offset_ = location;
        }
    }

    void advance(int bytes) override
    {
        code_generator_->add(reg_, bytes);
        offset_ += bytes;
    }

    void restore() override
    {
        if (offset_ != 0)
        {
            code_generator_->sub(reg_, offset_);
            offset_ = 0;
        }
    }

    Xbyak::RegExp get_address(int global_location) override
    {
        return reg_ + (global_location - offset_);
    }

    Xbyak::RegExp get_address_without_index(int global_location) override
    {
        return reg_ + (global_location - offset_);
    }
};

class null_address_packer : public address_packer
{
private:
    Xbyak::Reg64 reg_;

public:
    null_address_packer(Xbyak::CodeGenerator*, Xbyak::Reg64 reg)
        : reg_(reg)
    {
    }

    void loop_prologue() override {}

    void initialize() override {}

    void move_to(int) override {}

    void advance(int) override {}

    void restore() override {}

    Xbyak::RegExp get_address(int global_location) override
    {
        return reg_ + global_location;
    }

    Xbyak::RegExp get_address_without_index(int global_location) override
    {
        return reg_ + global_location;
    }
};

// Problem we are trying to solve here is Maximum coverage problem
// https://en.wikipedia.org/wiki/Maximum_coverage_problem We will use
// the proposed heuristic to achieve an approximation ratio of 1-1/e.

// We gonna start with a simple one for now (not take into account
// small deltas) or what kind of addressing types we are using (ptr vs
// ptr_b, avx vs avx512).

class simple_SIB_address_packer : public address_packer
{
private:
    Xbyak::CodeGenerator*       code_generator_;
    Xbyak::Reg64                reg_;
    std::map<int, Xbyak::Reg64> indices_;
    int                         offset_ = 0;

public:
    simple_SIB_address_packer(Xbyak::CodeGenerator* code_generator,
                              Xbyak::Reg64 reg, int N, int delta,
                              std::vector<Xbyak::Reg64> const& registers)
        : code_generator_(code_generator)
        , reg_(reg)
    {
        std::vector<int> covered(N + 1);
        std::set<int>    selected;

        int K = registers.size();

        for (int k = 0; k < K; ++k)
        {
            int best   = 0;
            int covers = 0;

            for (int i = 1; i <= N; ++i)
            {
                int ccovered = 0;
                for (int f = 1; f <= 8; f *= 2)
                {
                    if (i * f <= N)
                    {
                        if (covered[i * f] == 0)
                        {
                            ++ccovered;
                        }
                    }
                }

                if (ccovered > covers)
                {
                    covers = ccovered;
                    best   = i;
                }
            }

            if (covers == 0)
            {
                break;
            }

            indices_[best * delta] = registers[k];

            for (int f = 1; f <= 8; f *= 2)
            {
                if (f * best <= N)
                {
                    ++covered[f * best];
                }
            }
        }
    }

    void loop_prologue() override {}

    void initialize() override
    {
        for (auto const& p : indices_)
        {
            code_generator_->mov(p.second, p.first);
        }
    }

    void move_to(int location) override
    {
        if (offset_ != location)
        {
            code_generator_->add(reg_, (location - offset_));
            offset_ = location;
        }
    }

    void advance(int bytes) override
    {
        code_generator_->add(reg_, bytes);
        offset_ += bytes;
    }

    void restore() override
    {
        if (offset_ != 0)
        {
            code_generator_->sub(reg_, offset_);
            offset_ = 0;
        }
    }

    Xbyak::RegExp get_address(int global_location) override
    {
        int           local_location = global_location - offset_;
        Xbyak::RegExp best           = reg_ + local_location;

        int best_distance = local_location;

        for (auto const& p : indices_)
        {
            for (int f = 1; f <= 8; f *= 2)
            {
                int loc = p.first * f;
                if (std::abs(loc - local_location) < best_distance)
                {
                    best_distance = std::abs(loc - local_location);
                    best = reg_ + p.second * f + (local_location - loc);
                }
            }
        }

        return best;
    }

    Xbyak::RegExp get_address_without_index(int global_location) override
    {
        int local_location = global_location - offset_;
        return reg_ + local_location;
    }
};

class double_base_SIB_address_packer : public address_packer
{
private:
    Xbyak::CodeGenerator*       code_generator_;
    Xbyak::Reg64                reg_;
    std::map<int, Xbyak::Reg64> indices_;
    int                         offset_ = 0;
    Xbyak::Reg64                second_base_;
    int                         second_base_at_;

public:
    double_base_SIB_address_packer(Xbyak::CodeGenerator* code_generator,
                                   Xbyak::Reg64 reg, int N, int delta,
                                   std::vector<Xbyak::Reg64> const& registers)
        : code_generator_(code_generator)
        , reg_(reg)
    {
        std::vector<int> covered(N + 1);
        std::set<int>    selected;

        int K = registers.size();

        second_base_    = registers[0];
        second_base_at_ = delta * (N + 1) / 2;

        for (int k = 1; k < K; ++k)
        {
            int best   = 0;
            int covers = 0;

            for (int i = 1; i <= (N + 1) / 2 - 1; ++i)
            {
                int ccovered = 0;
                for (int f = 1; f <= 8; f *= 2)
                {
                    if (i * f <= N)
                    {
                        if (covered[i * f] == 0)
                        {
                            ++ccovered;
                        }
                    }
                }

                if (ccovered > covers)
                {
                    covers = ccovered;
                    best   = i;
                }
            }

            if (covers == 0)
            {
                break;
            }

            indices_[best * delta] = registers[k];

            for (int f = 1; f <= 8; f *= 2)
            {
                if (f * best <= N)
                {
                    ++covered[f * best];
                }
            }
        }
    }

    void loop_prologue() override
    {
        code_generator_->lea(second_base_,
                             code_generator_->ptr[reg_ + second_base_at_]);
    }

    void initialize() override
    {
        for (auto const& p : indices_)
        {
            code_generator_->mov(p.second, p.first);
        }
    }

    void move_to(int location) override
    {
        if (offset_ != location)
        {
            code_generator_->add(reg_, (location - offset_));
            code_generator_->add(second_base_, (location - offset_));
            offset_ = location;
        }
    }

    void advance(int bytes) override
    {
        code_generator_->add(reg_, bytes);
        code_generator_->add(second_base_, bytes);
        offset_ += bytes;
    }

    void restore() override
    {
        if (offset_ != 0)
        {
            code_generator_->sub(reg_, offset_);
            offset_ = 0;
        }
    }

    Xbyak::RegExp get_address(int global_location) override
    {
        int local_location = global_location - offset_;

        auto base_reg = reg_;

        if (local_location >= second_base_at_)
        {
            local_location -= second_base_at_;
            base_reg = second_base_;
        }

        Xbyak::RegExp best = base_reg + local_location;

        int best_distance = local_location;

        for (auto const& p : indices_)
        {
            for (int f = 1; f <= 8; f *= 2)
            {
                int loc = p.first * f;
                if (std::abs(loc - local_location) < best_distance)
                {
                    best_distance = std::abs(loc - local_location);
                    best = base_reg + p.second * f + (local_location - loc);
                }
            }
        }

        return best;
    }

    Xbyak::RegExp get_address_without_index(int global_location) override
    {
        int local_location = global_location - offset_;

        auto base_reg = reg_;

        if (local_location >= second_base_at_)
        {
            local_location -= second_base_at_;
            base_reg = second_base_;
        }

        return base_reg + local_location;
    }
};

} // namespace sysml
} // namespace facebook
