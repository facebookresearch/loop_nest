// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"

#include "common.h"
#include "isa.h"

#include <cstdint>
#include <limits>
#include <memory>

namespace facebook
{
namespace sysml
{
namespace aot
{

class basic_plus
{
public:
    void issue_epilogue(Xbyak::CodeGenerator& cg)
    {
        // do nothing
    }

    template <class RegType>
    void set_to_identity(Xbyak::CodeGenerator& cg, const RegType& dest) const
    {
        cg.vxorpd(dest, dest, dest);
    }

    template <class RegType>
    void issue(Xbyak::CodeGenerator& cg, const RegType& dest,
               const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        cg.vaddps(dest, op1, op2);
    }
};

// solely for purposes of testing non-fused multiply-accumulate
class duplicate_base_plus
{
public:
    void issue_epilogue(Xbyak::CodeGenerator& cg)
    {
        // do nothing
    }

    template <class RegType>
    void set_to_identity(Xbyak::CodeGenerator& cg, const RegType& dest) const
    {
        cg.vxorpd(dest, dest, dest);
    }

    template <class RegType>
    void issue(Xbyak::CodeGenerator& cg, const RegType& dest,
               const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        cg.vaddps(dest, op1, op2);
    }
};

class max
{
private:
    Xbyak::Label identity_label;

public:
    void issue_epilogue(Xbyak::CodeGenerator& cg)
    {
        std::uint32_t identity_value =
            bit_cast<std::uint32_t>(-std::numeric_limits<float>::infinity());
        cg.L(identity_label);
        cg.dd(identity_value);
    }

    template <class RegType>
    void set_to_identity(Xbyak::CodeGenerator& cg, const RegType& dest) const
    {
        cg.vbroadcastss(dest, cg.ptr[cg.rip + identity_label]);
    }

    template <class RegType>
    void issue(Xbyak::CodeGenerator& cg, const RegType& dest,
               const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        cg.vmaxps(dest, op1, op2);
    }
};

class min
{
private:
    Xbyak::Label identity_label;

public:
    void issue_epilogue(Xbyak::CodeGenerator& cg)
    {
        std::uint32_t identity_value =
            bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity());
        cg.L(identity_label);
        cg.dd(identity_value);
    }

    template <class RegType>
    void set_to_identity(Xbyak::CodeGenerator& cg, const RegType& dest) const
    {
        cg.vbroadcastss(dest, cg.ptr[cg.rip + identity_label]);
    }

    template <class RegType>
    void issue(Xbyak::CodeGenerator& cg, const RegType& dest,
               const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        cg.vminps(dest, op1, op2);
    }
};

class basic_multiplies
{

public:
    template <class RegType>
    void issue(Xbyak::CodeGenerator& cg, const RegType& dest,
               const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        cg.vmulps(dest, op1, op2);
    }
};

class operation_pair_base
{
public:
    virtual ~operation_pair_base() {}

    virtual void issue_epilogue(Xbyak::CodeGenerator& cg) = 0;

    virtual void set_to_identity(Xbyak::CodeGenerator& cg,
                                 const Xbyak::Xmm&     dest) const = 0;
    virtual void set_to_identity(Xbyak::CodeGenerator& cg,
                                 const Xbyak::Ymm&     dest) const = 0;
    virtual void set_to_identity(Xbyak::CodeGenerator& cg,
                                 const Xbyak::Zmm&     dest) const = 0;

    virtual void plus(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
                      const Xbyak::Operand& op1,
                      const Xbyak::Operand& op2) const = 0;
    virtual void plus(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
                      const Xbyak::Operand& op1,
                      const Xbyak::Operand& op2) const = 0;
    virtual void plus(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
                      const Xbyak::Operand& op1,
                      const Xbyak::Operand& op2) const = 0;

    virtual void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
                            const Xbyak::Operand& op1,
                            const Xbyak::Operand& op2) const = 0;
    virtual void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
                            const Xbyak::Operand& op1,
                            const Xbyak::Operand& op2) const = 0;
    virtual void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
                            const Xbyak::Operand& op1,
                            const Xbyak::Operand& op2) const = 0;

    virtual bool can_fuse() const = 0;

    virtual void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
                      const Xbyak::Xmm&     op1,
                      const Xbyak::Operand& op2) const = 0;
    virtual void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
                      const Xbyak::Ymm&     op1,
                      const Xbyak::Operand& op2) const = 0;
    virtual void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
                      const Xbyak::Zmm&     op1,
                      const Xbyak::Operand& op2) const = 0;
};

template <class PlusType, class MultipliesType>
class operation_pair : public operation_pair_base
{
private:
    PlusType       plus_op;
    MultipliesType multiplies_op;

private:
    template <class RegType>
    void set_to_identity_(Xbyak::CodeGenerator& cg, const RegType& dest) const
    {
        plus_op.set_to_identity(cg, dest);
    }

    template <class RegType>
    void plus_(Xbyak::CodeGenerator& cg, const RegType& dest,
               const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        plus_op.issue(cg, dest, op1, op2);
    }

    template <class RegType>
    void multiplies_(Xbyak::CodeGenerator& cg, const RegType& dest,
                     const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        multiplies_op.issue(cg, dest, op1, op2);
    }

    template <class RegType>
    void fuse_(Xbyak::CodeGenerator& cg, const RegType& dest,
               const RegType& op1, const Xbyak::Operand& op2) const
    {
        assert(false);
    }

public:
    void issue_epilogue(Xbyak::CodeGenerator& cg)
    {
        plus_op.issue_epilogue(cg);
    }

    void set_to_identity(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest) const
    {
        set_to_identity_(cg, dest);
    }

    void set_to_identity(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest) const
    {
        set_to_identity_(cg, dest);
    }

    void set_to_identity(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest) const
    {
        set_to_identity_(cg, dest);
    }

    void plus(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
              const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        plus_(cg, dest, op1, op2);
    }

    void plus(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
              const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        plus_(cg, dest, op1, op2);
    }

    void plus(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
              const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        plus_(cg, dest, op1, op2);
    }

    void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
                    const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        multiplies_(cg, dest, op1, op2);
    }

    void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
                    const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        multiplies_(cg, dest, op1, op2);
    }

    void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
                    const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        multiplies_(cg, dest, op1, op2);
    }

    bool can_fuse() const { return false; }

    void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
              const Xbyak::Xmm& op1, const Xbyak::Operand& op2) const
    {
        fuse_(cg, dest, op1, op2);
    }

    void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
              const Xbyak::Ymm& op1, const Xbyak::Operand& op2) const
    {
        fuse_(cg, dest, op1, op2);
    }

    void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
              const Xbyak::Zmm& op1, const Xbyak::Operand& op2) const
    {
        fuse_(cg, dest, op1, op2);
    }
};

template <>
class operation_pair<basic_plus, basic_multiplies> : public operation_pair_base
{
private:
    basic_plus       plus_op;
    basic_multiplies multiplies_op;

private:
    template <class RegType>
    void set_to_identity_(Xbyak::CodeGenerator& cg, const RegType& dest) const
    {
        plus_op.set_to_identity(cg, dest);
    }

    template <class RegType>
    void plus_(Xbyak::CodeGenerator& cg, const RegType& dest,
               const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        plus_op.issue(cg, dest, op1, op2);
    }

    template <class RegType>
    void multiplies_(Xbyak::CodeGenerator& cg, const RegType& dest,
                     const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        multiplies_op.issue(cg, dest, op1, op2);
    }

    template <class RegType>
    void fuse_(Xbyak::CodeGenerator& cg, const RegType& dest,
               const RegType& op1, const Xbyak::Operand& op2) const
    {
        cg.vfmadd231ps(dest, op1, op2);
    }

public:
    void issue_epilogue(Xbyak::CodeGenerator& cg)
    {
        plus_op.issue_epilogue(cg);
    }

    void set_to_identity(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest) const
    {
        set_to_identity_(cg, dest);
    }

    void set_to_identity(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest) const
    {
        set_to_identity_(cg, dest);
    }

    void set_to_identity(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest) const
    {
        set_to_identity_(cg, dest);
    }

    void plus(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
              const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        plus_(cg, dest, op1, op2);
    }

    void plus(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
              const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        plus_(cg, dest, op1, op2);
    }

    void plus(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
              const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        plus_(cg, dest, op1, op2);
    }

    void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
                    const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        multiplies_(cg, dest, op1, op2);
    }

    void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
                    const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        multiplies_(cg, dest, op1, op2);
    }
    void multiplies(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
                    const Xbyak::Operand& op1, const Xbyak::Operand& op2) const
    {
        multiplies_(cg, dest, op1, op2);
    }

    bool can_fuse() const { return true; }

    void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
              const Xbyak::Xmm& op1, const Xbyak::Operand& op2) const
    {
        fuse_(cg, dest, op1, op2);
    }

    void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Ymm& dest,
              const Xbyak::Ymm& op1, const Xbyak::Operand& op2) const
    {
        fuse_(cg, dest, op1, op2);
    }

    void fuse(Xbyak::CodeGenerator& cg, const Xbyak::Zmm& dest,
              const Xbyak::Zmm& op1, const Xbyak::Operand& op2) const
    {
        fuse_(cg, dest, op1, op2);
    }
};

inline std::shared_ptr<operation_pair_base> const fma =
    std::make_shared<operation_pair<basic_plus, basic_multiplies>>();

// exclusively here to test non-fused operations as base case
inline std::shared_ptr<operation_pair_base> const non_fused_ma =
    std::make_shared<operation_pair<duplicate_base_plus, basic_multiplies>>();

inline std::shared_ptr<operation_pair_base> const multiply_max =
    std::make_shared<operation_pair<max, basic_multiplies>>();

inline std::shared_ptr<operation_pair_base> const multiply_min =
    std::make_shared<operation_pair<min, basic_multiplies>>();

inline std::shared_ptr<operation_pair_base> const plus_max =
    std::make_shared<operation_pair<max, basic_plus>>();

} // namespace aot
} // namespace sysml
} // namespace facebook
