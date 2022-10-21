// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"
#ifdef DABUN_ARCH_X86_64

#include "dabun/common.hpp"
#include "dabun/x86/xbyak.hpp"

#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dabun
{
namespace x86
{

template <class ISA>
class elementwise_operation
{
private:
    static constexpr int vector_size = isa_traits<ISA>::vector_size;
    using memory_argument            = memory_argument_type<vector_size>;

public:
    virtual ~elementwise_operation(){};

    virtual std::string name() = 0;

    // set up information for followed tensors
    // called in loop_nest code_generator constructor (after that elementwise is
    // stateless)
    virtual void
    initialize(std::vector<std::map<std::string, int>> const&, // followed
                                                               // tensor strides
               std::vector<tensor_traits> const&, // followed tensor traits
               Xbyak::Label* // possible label for avx2 full mask
               ) = 0;

    // vectorized processing
    virtual void process_batch(
        Xbyak::CodeGenerator&,
        std::vector<std::pair<memory_argument, Xbyak::Zmm>> const&, // mem and
                                                                    // register
        std::vector<Xbyak::Zmm> const&,    // auxiliary vector registers
        std::vector<Xbyak::Opmask> const&, // auxiliary k registers
        access_kind,                       // C access kind (strided/packed)
        avx512,
        std::optional<Xbyak::Opmask> // tail mask register
    ) const = 0;

    virtual void
    process_batch(Xbyak::CodeGenerator&,
                  std::vector<std::pair<memory_argument,
                                        Xbyak::Ymm>> const&, // mem and register
                  std::vector<Xbyak::Ymm> const&, // auxiliary registers
                  access_kind, // C access kind (strided/packed)
                  avx2,
                  std::optional<Xbyak::Ymm> // tail maks register
    ) const = 0;

    // scalar processing
    virtual void process_batch(
        Xbyak::CodeGenerator&,
        std::vector<std::pair<memory_argument, Xbyak::Xmm>> const&, // mem and
                                                                    // register
        std::vector<Xbyak::Xmm> const&, // auxiliary registers
        access_kind,                    // C access kind
        avx512) const = 0;

    virtual void process_batch(
        Xbyak::CodeGenerator&,
        std::vector<std::pair<memory_argument, Xbyak::Xmm>> const&, // mem and
                                                                    // register
        std::vector<Xbyak::Xmm> const&, // auxiliary registers for constants
        access_kind,                    // C access kind
        avx2) const = 0;
};

template <class ISA>
class relu_elementwise_operation : public elementwise_operation<ISA>
{
private:
    static constexpr int vector_size = isa_traits<ISA>::vector_size;
    using memory_argument            = memory_argument_type<vector_size>;

public:
    std::string name() override { return "ReLu"; }

    void initialize(std::vector<std::map<std::string, int>> const& = {},
                    std::vector<tensor_traits> const&              = {},
                    Xbyak::Label* = nullptr) override
    {
        // no state in relu
    }

    void
    process_batch(Xbyak::CodeGenerator& cg,
                  std::vector<std::pair<memory_argument, Xbyak::Zmm>> const&
                                                 mems_and_regs,
                  std::vector<Xbyak::Zmm> const& auxiliary,
                  std::vector<Xbyak::Opmask> const&,
                  [[maybe_unused]] access_kind access, avx512,
                  std::optional<Xbyak::Opmask> = std::nullopt) const override
    {
        assert(access == VECTOR_PACKED || access == VECTOR_STRIDED);
        assert(auxiliary.size());
        int auxiliary_index = auxiliary[0].getIdx();
        assert(auxiliary_index >= 0 && auxiliary_index < 32);

        cg.vxorpd(auxiliary[0], auxiliary[0], auxiliary[0]);

        for (auto const& e : mems_and_regs)
        {
            cg.vmaxps(e.second, e.second, Xbyak::Zmm(auxiliary_index));
        }
    }

    void process_batch(
        Xbyak::CodeGenerator& cg,
        std::vector<std::pair<memory_argument, Xbyak::Ymm>> const&
                                       mems_and_regs,
        std::vector<Xbyak::Ymm> const& auxiliary,
        [[maybe_unused]] access_kind   access, avx2,
        std::optional<Xbyak::Ymm> /* tail_mask */ = std::nullopt) const override
    {
        assert(access == VECTOR_PACKED || access == VECTOR_STRIDED);
        assert(auxiliary.size());
        int auxiliary_index = auxiliary[0].getIdx();
        assert(auxiliary_index >= 0 && auxiliary_index < 16);

        cg.vxorpd(auxiliary[0], auxiliary[0], auxiliary[0]);

        for (auto const& e : mems_and_regs)
        {
            cg.vmaxps(e.second, e.second, Xbyak::Ymm(auxiliary_index));
        }
    }

    void
    process_batch(Xbyak::CodeGenerator& cg,
                  std::vector<std::pair<memory_argument, Xbyak::Xmm>> const&
                                                 mems_and_regs,
                  std::vector<Xbyak::Xmm> const& auxiliary,
                  [[maybe_unused]] access_kind   kind, avx512) const override
    {

        assert(kind == SCALAR);
        assert(auxiliary.size() > 0);
        assert(auxiliary[0].getIdx() < 16);
        assert(mems_and_regs.size() == 1);
        cg.xorpd(auxiliary[0], auxiliary[0]);

        cg.maxps(mems_and_regs[0].second, auxiliary[0]);
    }

    void
    process_batch(Xbyak::CodeGenerator& cg,
                  std::vector<std::pair<memory_argument, Xbyak::Xmm>> const&
                                                 mems_and_regs,
                  std::vector<Xbyak::Xmm> const& auxiliary, access_kind kind,
                  avx2) const override
    {
        assert(kind == SCALAR);
        process_batch(cg, mems_and_regs, auxiliary, kind, avx512());
    }
};

inline void push_ymms_to_stack(Xbyak::CodeGenerator&          cg,
                               std::vector<Xbyak::Ymm> const& regs)
{
    // store to the red zone
    // https://en.wikipedia.org/wiki/Red_zone_(computing)
    int n            = regs.size();
    int bytes_needed = n * 32;
    assert(bytes_needed <= 128);

    // adjust rsp to provide red zone below it
    cg.sub(cg.rsp, bytes_needed);

    for (int i = 0; i < n; i++)
    {
        int offset = (n - i - 1) * 32;
        cg.vmovups(cg.ptr[cg.rsp + offset], regs[i]);
    }
}

inline void pop_ymms_from_stack(Xbyak::CodeGenerator&          cg,
                                std::vector<Xbyak::Ymm> const& regs)
{
    // load from the red zone
    // https://en.wikipedia.org/wiki/Red_zone_(computing)
    int n            = regs.size();
    int bytes_needed = n * 32;

    for (int i = 0; i < n; i++)
    {
        int offset = (n - i - 1) * 32;
        cg.vmovups(regs[i], cg.ptr[cg.rsp + offset]);
    }

    cg.add(cg.rsp, bytes_needed);
}

// operation with a single followed tensor (e.g. adding another tensor,
// multiplying by other tensor)
using single_tensor_op =
    std::function<void(Xbyak::CodeGenerator&, const Xbyak::Xmm&,
                       const Xbyak::Operand&, const Xbyak::Operand&)>;

template <class ISA>
class single_tensor_elementwise_operation : public elementwise_operation<ISA>
{
private:
    static constexpr int vector_size = isa_traits<ISA>::vector_size;
    using memory_argument            = memory_argument_type<vector_size>;

    std::vector<std::map<std::string, int>> t_strides;
    std::vector<tensor_traits>              t_traits;
    Xbyak::Label*                           avx2_full_mask;

    // defined once for the shared pointer
    std::string      op_name;
    single_tensor_op vector_op;
    single_tensor_op scalar_op;

public:
    single_tensor_elementwise_operation(std::string      op_name,
                                        single_tensor_op vector_op,
                                        single_tensor_op scalar_op)
        : op_name(op_name)
        , vector_op(vector_op)
        , scalar_op(scalar_op)
    {
    }

    std::string name() override { return op_name; }

    void initialize(std::vector<std::map<std::string, int>> const& t_strides_,
                    std::vector<tensor_traits> const&              t_traits_,
                    Xbyak::Label* avx2_full_mask_) override
    {
        t_strides      = t_strides_;
        t_traits       = t_traits_;
        avx2_full_mask = avx2_full_mask_;

        assert(t_strides.size());
        assert(t_traits.size());
    }

    void process_batch(
        Xbyak::CodeGenerator& cg,
        std::vector<std::pair<memory_argument, Xbyak::Zmm>> const&
                                          mems_and_regs,
        std::vector<Xbyak::Zmm> const&    auxillaries,
        std::vector<Xbyak::Opmask> const& in_kregs,
        [[maybe_unused]] access_kind      C_access, avx512,
        std::optional<Xbyak::Opmask> tail_mask = std::nullopt) const override
    {
        auto auxiliary = auxillaries;
        auto kregs     = in_kregs;
        assert(C_access == VECTOR_PACKED || C_access == VECTOR_STRIDED);
        assert(auxiliary.size());

        tensor_traits other_traits               = t_traits[0];
        access_kind   other_access               = other_traits.access;
        Xbyak::Reg64  other_addr_reg             = other_traits.reg;
        Xbyak::Label* other_stride_label         = other_traits.stridesLabel;
        std::map<std::string, int> other_strides = t_strides[0];

        Xbyak::Zmm other_data_reg = auxiliary.back();
        auxiliary.pop_back();

        // if other vector is strided
        Xbyak::Zmm arg_other_strides;

        // set by caller, since needs info on C and iterations
        Xbyak::Opmask tail_k_mask;
        if (tail_mask)
        {
            tail_k_mask = *tail_mask;
        }
        // set by us for other data if needed
        Xbyak::Opmask full_k_mask;
        Xbyak::Opmask temp_k_mask;

        if (other_access == VECTOR_STRIDED)
        {
            assert(auxiliary.size());
            arg_other_strides = auxiliary.back();
            auxiliary.pop_back();
            cg.vmovups(arg_other_strides,
                       cg.ptr[cg.rip + (*other_stride_label)]);

            assert(kregs.size() >= 2);

            temp_k_mask = kregs.back();
            kregs.pop_back();

            full_k_mask = kregs.back();
            kregs.pop_back();

            cg.mov(cg.r12, (1 << vector_size) - 1);
            cg.kmovw(full_k_mask, cg.r12.cvt32());
        }

        for (auto const& e : mems_and_regs)
        {
            memory_argument c_arg      = e.first;
            Xbyak::Ymm      c_data_reg = e.second;
            int             other_offset =
                get_cursor_offset(c_arg.coordinates, other_strides);

            // load data for followed other tensor
            switch (other_access)
            {
            case VECTOR_PACKED:
                if (c_arg.mask == vector_size)
                {
                    cg.vmovups(other_data_reg,
                               cg.ptr[other_addr_reg + other_offset * 4]);
                }
                else
                {
                    cg.vmovups(other_data_reg | tail_k_mask,
                               cg.ptr[other_addr_reg + other_offset * 4]);
                }
                break;

            case VECTOR_STRIDED:
                cg.kmovw(
                    temp_k_mask, // The mask gets updated in gather
                    (c_arg.mask == vector_size ? full_k_mask : tail_k_mask));
                cg.vgatherdps(other_data_reg | temp_k_mask,
                              cg.ptr[other_addr_reg + other_offset * 4 +
                                     arg_other_strides]);
                break;

            case SCALAR:
                cg.vbroadcastss(other_data_reg,
                                cg.ptr[other_addr_reg + other_offset * 4]);
                break;
            }

            vector_op(cg, c_data_reg, c_data_reg, other_data_reg);
        }
    }

    void process_batch(
        Xbyak::CodeGenerator& cg,
        std::vector<std::pair<memory_argument, Xbyak::Ymm>> const&
                                       mems_and_regs,
        std::vector<Xbyak::Ymm> const& auxiliary,
        [[maybe_unused]] access_kind   C_access, avx2,
        std::optional<Xbyak::Ymm>      tail_mask = std::nullopt) const override
    {

        assert(C_access == VECTOR_PACKED || C_access == VECTOR_STRIDED);
        assert(auxiliary.size());

        tensor_traits other_traits               = t_traits[0];
        access_kind   other_access               = other_traits.access;
        Xbyak::Reg64  other_addr_reg             = other_traits.reg;
        Xbyak::Label* other_stride_label         = other_traits.stridesLabel;
        std::map<std::string, int> other_strides = t_strides[0];

        std::vector<Xbyak::Ymm> remaining_auxiliary_ymm = auxiliary;

        Xbyak::Ymm other_data_reg = remaining_auxiliary_ymm.back();
        remaining_auxiliary_ymm.pop_back();

        // set by caller since need knowledge of C and iteration
        Xbyak::Ymm ymm_tail_mask;
        if (tail_mask)
        {
            ymm_tail_mask = *tail_mask;
        }
        // potentially used if strided followed tensor in elementwise
        Xbyak::Ymm ymm_full_mask;
        Xbyak::Ymm ymm_temp_mask;
        Xbyak::Ymm arg_other_strides;

        std::vector<Xbyak::Ymm> spilled_registers;

        if (other_access == VECTOR_STRIDED)
        {
            // we need to find 3 vector registers
            // for use as full mask, temp mask, and strided
            int extra_needed = 3;
            // start out with any left over registers provided by the user
            extra_needed -= remaining_auxiliary_ymm.size();
            // if we need extra, we'll have to spill
            std::vector<Xbyak::Ymm> extra_registers;

            if (extra_needed > 0)
            {
                int              occupied       = -1;
                std::vector<int> avx2_registers = {
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

                // registers provided by the user are already accounted for
                // so can't use those
                for (auto const& e : auxiliary)
                {
                    avx2_registers[e.getIdx()] = occupied;
                }

                // if we're using a tail mask, we can't use that one either
                if (tail_mask)
                {
                    avx2_registers[(*tail_mask).getIdx()] = occupied;
                }

                // can't use registers blocked for output tensor
                for (auto const& e : mems_and_regs)
                {
                    avx2_registers[e.second.getIdx()] = occupied;
                }

                // registers that are "free" and can be used (with spilling)
                for (auto const& r : avx2_registers)
                {
                    if (r != occupied && extra_registers.size() != extra_needed)
                    {
                        Xbyak::Ymm spilled = Xbyak::Ymm(r);
                        extra_registers.push_back(spilled);
                    }
                }

                assert(extra_registers.size() == extra_needed);
            }

            // extra registers need to be spilled to ensure state is maintained
            spilled_registers = extra_registers;
            push_ymms_to_stack(cg, spilled_registers);

            remaining_auxiliary_ymm.insert(remaining_auxiliary_ymm.end(),
                                           extra_registers.begin(),
                                           extra_registers.end());

            ymm_full_mask = remaining_auxiliary_ymm.back();
            remaining_auxiliary_ymm.pop_back();

            ymm_temp_mask = remaining_auxiliary_ymm.back();
            remaining_auxiliary_ymm.pop_back();

            arg_other_strides = remaining_auxiliary_ymm.back();
            remaining_auxiliary_ymm.pop_back();

            cg.vmovups(ymm_full_mask, cg.ptr[cg.rip + (*avx2_full_mask)]);
            cg.vmovups(arg_other_strides,
                       cg.ptr[cg.rip + (*other_stride_label)]);
        }

        for (auto const& e : mems_and_regs)
        {
            memory_argument c_arg      = e.first;
            Xbyak::Ymm      c_data_reg = e.second;
            int             other_offset =
                get_cursor_offset(c_arg.coordinates, other_strides);

            switch (other_access)
            {

            case VECTOR_PACKED:
                if (c_arg.mask == vector_size)
                {
                    cg.vmovups(other_data_reg,
                               cg.ptr[other_addr_reg + other_offset * 4]);
                }
                else
                {
                    cg.vmaskmovps(other_data_reg, ymm_tail_mask,
                                  cg.ptr[other_addr_reg + other_offset * 4]);
                }
                break;

            case VECTOR_STRIDED:
                cg.vmovups(ymm_temp_mask,
                           (c_arg.mask == vector_size ? ymm_full_mask
                                                      : ymm_tail_mask));
                cg.vgatherdps(other_data_reg,
                              cg.ptr[other_addr_reg + other_offset * 4 +
                                     arg_other_strides],
                              ymm_temp_mask);
                break;

            case SCALAR:
                cg.vbroadcastss(other_data_reg,
                                cg.ptr[other_addr_reg + other_offset * 4]);
                break;
            }

            vector_op(cg, c_data_reg, c_data_reg, other_data_reg);
        }

        if (spilled_registers.size())
        {
            pop_ymms_from_stack(cg, spilled_registers);
        }
    }

    void
    process_batch(Xbyak::CodeGenerator& cg,
                  std::vector<std::pair<memory_argument, Xbyak::Xmm>> const&
                      mems_and_regs,
                  std::vector<Xbyak::Xmm> const& /* auxiliary */,
                  [[maybe_unused]] access_kind C_access, avx512) const override
    {
        tensor_traits                other_traits   = t_traits[0];
        [[maybe_unused]] access_kind other_access   = other_traits.access;
        Xbyak::Reg64                 other_addr_reg = other_traits.reg;
        std::map<std::string, int>   other_strides  = t_strides[0];

        assert(C_access == SCALAR && other_access == SCALAR);

        for (auto const& e : mems_and_regs)
        {
            memory_argument c_arg      = e.first;
            Xbyak::Xmm      c_data_reg = e.second;
            int             other_offset =
                get_cursor_offset(c_arg.coordinates, other_strides);

            scalar_op(cg, c_data_reg, c_data_reg,
                      cg.ptr[other_addr_reg + other_offset * 4]);
        }
    }

    void
    process_batch(Xbyak::CodeGenerator& cg,
                  std::vector<std::pair<memory_argument, Xbyak::Xmm>> const&
                      mems_and_regs,
                  std::vector<Xbyak::Xmm> const& /* auxiliary */,
                  [[maybe_unused]] access_kind C_access, avx2) const override
    {
        tensor_traits                other_traits   = t_traits[0];
        [[maybe_unused]] access_kind other_access   = other_traits.access;
        Xbyak::Reg64                 other_addr_reg = other_traits.reg;
        std::map<std::string, int>   other_strides  = t_strides[0];

        assert(C_access == SCALAR && other_access == SCALAR);

        for (auto const& e : mems_and_regs)
        {
            memory_argument c_arg      = e.first;
            Xbyak::Xmm      c_data_reg = e.second;
            int             other_offset =
                get_cursor_offset(c_arg.coordinates, other_strides);

            scalar_op(cg, c_data_reg, c_data_reg,
                      cg.ptr[other_addr_reg + other_offset * 4]);
        }
    }
};

template <class ISA>
inline std::shared_ptr<elementwise_operation<ISA>> const
    elementwise_relu = std::make_shared<relu_elementwise_operation<ISA>>();

template <class ISA>
inline std::shared_ptr<elementwise_operation<ISA>> const elementwise_bias =
    std::make_shared<single_tensor_elementwise_operation<ISA>>(
        "Add Bias",
        [](Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
           const Xbyak::Operand& o1, const Xbyak::Operand& o2)
        { cg.vaddps(dest, o1, o2); },
        [](Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
           const Xbyak::Operand& o1, const Xbyak::Operand& o2)
        { cg.vaddss(dest, o1, o2); });

template <class ISA>
inline std::shared_ptr<elementwise_operation<ISA>> const elementwise_multiply =
    std::make_shared<single_tensor_elementwise_operation<ISA>>(
        "Multiply by Other",
        [](Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
           const Xbyak::Operand& o1, const Xbyak::Operand& o2)
        { cg.vmulps(dest, o1, o2); },
        [](Xbyak::CodeGenerator& cg, const Xbyak::Xmm& dest,
           const Xbyak::Operand& o1, const Xbyak::Operand& o2)
        { cg.vmulss(dest, o1, o2); });

template <class ISA, class... ISAs>
class composed_elementwise : public elementwise_operation<ISA>
{
private:
    static_assert((std::is_same_v<ISA, ISAs> && ...));
    std::vector<std::shared_ptr<elementwise_operation<ISA>>> operations;
    std::string                                              op_name;

private:
    static constexpr int vector_size = isa_traits<ISA>::vector_size;
    using memory_argument            = memory_argument_type<vector_size>;

public:
    composed_elementwise(
        std::shared_ptr<elementwise_operation<ISA>> const& first,
        std::shared_ptr<elementwise_operation<ISAs>> const&... rest)
        : operations({first, rest...})
    {
        std::ostringstream oss;
        assert(operations.size());
        oss << operations[0]->name();
        for (std::size_t i = 0; i < operations.size(); ++i)
        {
            oss << "," << operations[i]->name();
        }
        op_name = oss.str();
    }

    std::string name() override { return op_name; }

    void initialize(std::vector<std::map<std::string, int>> const& map = {},
                    std::vector<tensor_traits> const& tensor_traits    = {},
                    Xbyak::Label*                     label = nullptr) override
    {
        for (auto& op : operations)
        {
            op->initialize(map, tensor_traits, label);
        }
    }

    void process_batch(
        Xbyak::CodeGenerator& cg,
        std::vector<std::pair<memory_argument, Xbyak::Zmm>> const&
                                          mems_and_regs,
        std::vector<Xbyak::Zmm> const&    auxiliary,
        std::vector<Xbyak::Opmask> const& aux_kregs, access_kind access, avx512,
        std::optional<Xbyak::Opmask> tail_k_mask = std::nullopt) const override
    {
        for (auto& op : operations)
        {
            op->process_batch(cg, mems_and_regs, auxiliary, aux_kregs, access,
                              avx512(), tail_k_mask);
        }
    }

    void process_batch(
        Xbyak::CodeGenerator& cg,
        std::vector<std::pair<memory_argument, Xbyak::Ymm>> const&
                                       mems_and_regs,
        std::vector<Xbyak::Ymm> const& auxiliary, access_kind access, avx2,
        std::optional<Xbyak::Ymm> tail_mask = std::nullopt) const override
    {
        for (auto& op : operations)
        {
            op->process_batch(cg, mems_and_regs, auxiliary, access, avx2(),
                              tail_mask);
        }
    }

    void
    process_batch(Xbyak::CodeGenerator& cg,
                  std::vector<std::pair<memory_argument, Xbyak::Xmm>> const&
                                                 mems_and_regs,
                  std::vector<Xbyak::Xmm> const& auxiliary, access_kind kind,
                  avx512) const override
    {
        for (auto& op : operations)
        {
            op->process_batch(cg, mems_and_regs, auxiliary, kind, avx512());
        }
    }

    void
    process_batch(Xbyak::CodeGenerator& cg,
                  std::vector<std::pair<memory_argument, Xbyak::Xmm>> const&
                                                 mems_and_regs,
                  std::vector<Xbyak::Xmm> const& auxiliary, access_kind kind,
                  avx2) const override
    {
        for (auto& op : operations)
        {
            op->process_batch(cg, mems_and_regs, auxiliary, kind, avx2());
        }
    }
};

template <class ISA, class... ISAs>
inline std::shared_ptr<elementwise_operation<ISA>>
compose(std::shared_ptr<elementwise_operation<ISA>> const& first,
        std::shared_ptr<elementwise_operation<ISAs>> const&... rest)
{
    static_assert((std::is_same_v<ISA, ISAs> && ...));
    return std::make_shared<composed_elementwise<ISA, ISAs...>>(first, rest...);
}

} // namespace x86
} // namespace dabun

#endif
