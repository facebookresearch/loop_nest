// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/isa.hpp"
#ifdef DABUN_ARCH_AARCH64

#include "dabun/code_generator/xbyak.hpp"
#include "dabun/core.hpp"

namespace dabun
{
namespace arm
{

template <class CodeGenerator>
class meta_mnemonics
{
private:
    Xbyak_aarch64::XReg const stack_register;
    Xbyak_aarch64::XReg const tmp_register;

private:
    CodeGenerator& self() { return static_cast<CodeGenerator&>(*this); }

public:
    meta_mnemonics(Xbyak_aarch64::XReg const& s, Xbyak_aarch64::XReg const& t)
        : stack_register(s)
        , tmp_register(t)
    {
    }

    template <class T>
    void meta_add_or_sub_imm(Xbyak_aarch64::XReg const& srcdst, T imm,
                             Xbyak_aarch64::XReg const& tmpreg)
    {
        strong_assert(srcdst.getIdx() != tmpreg.getIdx());

        if (imm == 0)
            return;

        if (imm > 0)
        {
            self().add_imm(srcdst, srcdst, imm, tmpreg);
        }
        else
        {
            self().sub_imm(srcdst, srcdst, -imm, tmpreg);
        }
    }

    template <class VectorRegister>
    void meta_ldr_post_ptr(VectorRegister const&      vr,
                           Xbyak_aarch64::XReg const& addr, int delta,
                           Xbyak_aarch64::XReg const& tmpreg)
    {
        if (delta && delta < 256 && delta >= -256)
        {
            self().ldr(vr, Xbyak_aarch64::post_ptr(addr, delta));
        }
        else
        {
            self().ldr(vr, Xbyak_aarch64::ptr(addr));
            meta_add_or_sub_imm(addr, delta, tmpreg);
        }
    }

    template <class VectorRegister>
    void meta_str_post_ptr(VectorRegister const&      vr,
                           Xbyak_aarch64::XReg const& addr, int delta,
                           Xbyak_aarch64::XReg const& tmpreg)
    {
        if (delta && delta < 256 && delta >= -256)
        {
            self().str(vr, Xbyak_aarch64::post_ptr(addr, delta));
        }
        else
        {
            self().str(vr, Xbyak_aarch64::ptr(addr));
            meta_add_or_sub_imm(addr, delta, tmpreg);
        }
    }

    template <class VectorRegister>
    void meta_ldrh_post_ptr(VectorRegister const&      vr,
                            Xbyak_aarch64::XReg const& addr, int delta,
                            Xbyak_aarch64::XReg const& tmpreg)
    {
        if (delta && delta < 256 && delta >= -256)
        {
            self().ldrh(vr, Xbyak_aarch64::post_ptr(addr, delta));
        }
        else
        {
            self().ldrh(vr, Xbyak_aarch64::ptr(addr));
            meta_add_or_sub_imm(addr, delta, tmpreg);
        }
    }

    template <class VectorRegister>
    void meta_strh_post_ptr(VectorRegister const&      vr,
                            Xbyak_aarch64::XReg const& addr, int delta,
                            Xbyak_aarch64::XReg const& tmpreg)
    {
        if (delta && delta < 256 && delta >= -256)
        {
            self().strh(vr, Xbyak_aarch64::post_ptr(addr, delta));
        }
        else
        {
            self().strh(vr, Xbyak_aarch64::ptr(addr));
            meta_add_or_sub_imm(addr, delta, tmpreg);
        }
    }

    template <class VectorRegister>
    void meta_ldp_post_ptr(VectorRegister const& vr1, VectorRegister const& vr2,
                           Xbyak_aarch64::XReg const& addr, int delta,
                           Xbyak_aarch64::XReg const& tmpreg)
    {
        int num_32bit_elements = vr1.getBit() / 32;

        strong_assert(num_32bit_elements > 0);

        if (delta && delta <= 252 * num_32bit_elements &&
            delta >= -256 * num_32bit_elements &&
            (delta % (4 /* bytes */ * num_32bit_elements) == 0))
        {
            self().ldp(vr1, vr2, Xbyak_aarch64::post_ptr(addr, delta));
        }
        else
        {
            self().ldp(vr1, vr2, Xbyak_aarch64::ptr(addr));
            meta_add_or_sub_imm(addr, delta, tmpreg);
        }
    }

    template <class VectorRegister>
    void meta_stp_post_ptr(VectorRegister const& vr1, VectorRegister const& vr2,
                           Xbyak_aarch64::XReg const& addr, int delta,
                           Xbyak_aarch64::XReg const& tmpreg)
    {
        int num_32bit_elements = vr1.getBit() / 32;

        strong_assert(num_32bit_elements > 0);

        if (delta && delta <= 252 * num_32bit_elements &&
            delta >= -256 * num_32bit_elements &&
            (delta % (4 /* bytes */ * num_32bit_elements) == 0))
        {
            self().stp(vr1, vr2, Xbyak_aarch64::post_ptr(addr, delta));
        }
        else
        {
            self().stp(vr1, vr2, Xbyak_aarch64::ptr(addr));
            meta_add_or_sub_imm(addr, delta, tmpreg);
        }
    }

    template <class VectorRegister>
    void meta_ldr_post_ptr(VectorRegister const&      vr,
                           Xbyak_aarch64::XReg const& addr, int delta)
    {
        meta_ldr_post_ptr(vr, addr, delta, tmp_register);
    }

    template <class VectorRegister>
    void meta_ldrh_post_ptr(VectorRegister const&      vr,
                            Xbyak_aarch64::XReg const& addr, int delta)
    {
        meta_ldrh_post_ptr(vr, addr, delta, tmp_register);
    }

    template <class VectorRegister>
    void meta_ldp_post_ptr(VectorRegister const& vr1, VectorRegister const& vr2,
                           Xbyak_aarch64::XReg const& addr, int delta)
    {
        meta_ldp_post_ptr(vr1, vr2, addr, delta, tmp_register);
    }

    template <class VectorRegister>
    void meta_str_post_ptr(VectorRegister const&      vr,
                           Xbyak_aarch64::XReg const& addr, int delta)
    {
        meta_str_post_ptr(vr, addr, delta, tmp_register);
    }

    template <class VectorRegister>
    void meta_strh_post_ptr(VectorRegister const&      vr,
                            Xbyak_aarch64::XReg const& addr, int delta)
    {
        meta_strh_post_ptr(vr, addr, delta, tmp_register);
    }

    template <class VectorRegister>
    void meta_stp_post_ptr(VectorRegister const& vr1, VectorRegister const& vr2,
                           Xbyak_aarch64::XReg const& addr, int delta)
    {
        meta_stp_post_ptr(vr1, vr2, addr, delta, tmp_register);
    }

    template <class T>
    void meta_sub_imm(Xbyak_aarch64::XReg const& srcdst, T imm)
    {
        if (imm == 0)
            return;

        self().sub_imm(srcdst, srcdst, imm, tmp_register);
    }

    template <class T>
    void meta_add_imm(Xbyak_aarch64::XReg const& srcdst, T imm)
    {
        if (imm == 0)
            return;

        self().add_imm(srcdst, srcdst, imm, tmp_register);
    }

    template <class T>
    void meta_sadd_imm(Xbyak_aarch64::XReg const& srcdst, T imm)
    {
        self().meta_add_or_sub_imm(srcdst, imm, tmp_register);
    }

    void meta_cmp(Xbyak_aarch64::XReg const& xreg, int imm)
    {
        if (imm >= -256 && imm < 256)
        {
            self().cmp(xreg, imm);
        }
        else
        {
            strong_assert((imm & 0xffff) ==
                          imm); // Add support for larger values
            // meta_mov_imm(tmp_register, imm);
            self().mov(tmp_register, imm);
            self().cmp(xreg, tmp_register);
        }
    }

    template <typename T>
    void meta_mov_imm(const Xbyak_aarch64::XReg& dst, T imm,
                      const Xbyak_aarch64::XReg& tmp)
    {
        strong_assert(dst.getIdx() != tmp.getIdx());

        int64_t  bit_ptn = static_cast<int64_t>(imm);
        uint64_t mask    = 0xFFFF;
        bool     flag    = false;

        /* ADD(immediate) supports unsigned imm12 */
        const uint64_t IMM12_MASK = ~uint64_t(0xfff);
        if ((bit_ptn & IMM12_MASK) == 0)
        { // <= 4095
            self().mov(dst, static_cast<uint32_t>(imm & 0xfff));
            return;
        }

        /* MOVZ allows shift amount = 0, 16, 32, 48 */
        for (int i = 0; i < 64; i += 16)
        {
            uint64_t tmp_ptn = (bit_ptn & (mask << i)) >> i;
            if (tmp_ptn)
            {
                if (!flag)
                {
                    self().movz(dst, static_cast<uint32_t>(tmp_ptn), i);
                    flag = true;
                }
                else
                {
                    self().movz(tmp, static_cast<uint32_t>(tmp_ptn), i);
                    self().add(dst, dst, tmp);
                }
            }
        }

        return;
    }

    template <typename T>
    void meta_mov_imm(const Xbyak_aarch64::XReg& dst, T imm)
    {
        strong_assert(dst.getIdx() != tmp_register.getIdx());
        meta_mov_imm(dst, imm, tmp_register);
    }

    // https://stackoverflow.com/questions/27941220/push-lr-and-pop-lr-in-arm-arch64
    void meta_push(Xbyak_aarch64::XReg const& op)
    {
        self().str(op, Xbyak_aarch64::post_ptr(stack_register, 8));
    }

    void meta_pop(Xbyak_aarch64::XReg const& op)
    {
        self().ldr(op, Xbyak_aarch64::pre_ptr(stack_register, -8));
    }

    void meta_push_pair(Xbyak_aarch64::XReg const& op1,
                        Xbyak_aarch64::XReg const& op2)
    {
        self().stp(op1, op2, Xbyak_aarch64::post_ptr(stack_register, 16));
    }

    void meta_pop_pair(Xbyak_aarch64::XReg const& op1,
                       Xbyak_aarch64::XReg const& op2)
    {
        self().ldp(op1, op2, Xbyak_aarch64::pre_ptr(stack_register, -16));
    }

    void meta_push(std::vector<Xbyak_aarch64::XReg> const& regs)
    {
        for (int i = 1; i < regs.size(); i += 2)
        {
            meta_push_pair(regs[i - 1], regs[i]);
        }

        if (regs.size() % 2)
        {
            meta_push(regs.back());
        }
    }

    void meta_pop(std::vector<Xbyak_aarch64::XReg> const& regs)
    {
        if (regs.size() % 2)
        {
            meta_pop(regs.back());
        }

        for (int i = static_cast<int>(regs.size() - (regs.size() % 2) - 2);
             i >= 0; i -= 2)
        {
            meta_pop_pair(regs[i], regs[i + 1]);
        }
    }
};

} // namespace arm
} // namespace dabun

#endif
