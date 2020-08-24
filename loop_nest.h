// TODO(partially done) - generalize to FN(alpha C + A * B + W)
// -- needs the W

// TODO - arbitrary inner operation

// TODO(partially done) - Better tails (try to reuse the fully unrolled code)

// TODO(partially done) - document the logic for relu (with the +1
// thing) We are using the upper 31 bits of AlphaReg store the sum of
// all visited dimensions

#pragma once

#if defined(LOOP_NEST_ARM) || defined(ARM_LOOP_NEST)
#include "arm_loop_nest.h"
#else

#include "address_packer.h"
#include "arithmetic_operation.h"
#include "code_generator.h"
#include "common.h"
#include "configuration.h"
#include "denormals.h"
#include "elementwise_operation.h"
#include "isa.h"
#include "log.h"
#include "math.h"
#include "most_frequent_queue.h"
#include "multi_vmm.h"
#include "utils.h"

#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>

#include <cstdint>
#include <deque>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace facebook
{
namespace sysml
{
namespace aot
{

template <class ISA>
class FMA_loop_nest_jitter
    : public code_generator<void(float* C, float const* A, float const* B,
                                 int alpha)>
{
private:
    using base =
        code_generator<void(float* C, float const* A, float const* B, int)>;
    using Vmm       = std::conditional_t<std::is_same_v<ISA, avx512>, Zmm, Ymm>;
    using multi_zmm = multi_vmm<Vmm>;

    static constexpr int vector_size = isa_traits<ISA>::vector_size;

    int stack_offset = 0;

    int push(Xbyak::Operand const& op)
    {
        base::push(op);
        return stack_offset++;
    }

    int push(std::uint32_t imm)
    {
        base::push(imm);
        return stack_offset++;
    }

    int push(Xbyak::AddressFrame const& af, std::uint32_t imm)
    {
        base::push(af, imm);
        return stack_offset++;
    }

    void pop(Xbyak::Operand const& op)
    {
        base::pop(op);
        --stack_offset;
    }

    void push(std::vector<Reg64> const& regs)
    {
        for (auto const& r : regs)
        {
            push(r);
        }
    }

    void pop(std::vector<Reg64> const& regs)
    {
        for (auto it = regs.crbegin(); it != regs.crend(); ++it)
        {
            pop(*it);
        }
    }

    auto at_stack_offset(int off) { return rsp + (stack_offset - off) * 8; }

    struct tensor_location_t
    {
        int idx;
        int offset;

        friend bool operator<(tensor_location_t const& lhs,
                              tensor_location_t const& rhs)
        {
            return std::tie(lhs.idx, lhs.offset) <
                   std::tie(rhs.idx, rhs.offset);
        }
    };

    struct memory_src
    {
        int  idx;
        int  offset;
        bool broadcast;
    };

    struct fmla_instruction
    {
        int                           dst;
        int                           left_src;
        std::variant<int, memory_src> right_src;

        friend bool operator<(fmla_instruction const& lhs,
                              fmla_instruction const& rhs)
        {
            return std::tie(lhs.dst, lhs.left_src, lhs.right_src) <
                   std::tie(rhs.dst, rhs.left_src, rhs.right_src);
        }
    };

    struct load_instruction
    {
        int vmm_idx;
        int does_broadcast;

        tensor_location_t tensor_loc;

        friend bool operator<(load_instruction const& lhs,
                              load_instruction const& rhs)
        {
            return std::tie(lhs.vreg, lhs.num_lanes, lhs.tensor_location) <
                   std::tie(rhs.vreg, rhs.num_lanes, rhs.tensor_location);
        }
    };

    using instruction_t = std::variant<load_instruction, fmla_instruction>;

    std::deque<std::vector<instruction_t>> instruction_IRs;

    std::map<int, std::vector<int>> load_offsets;
    bool                            load_offsets_first_pass = true;

private:
    Reg64              CReg_     = rdi;
    Reg64              AReg_     = rsi;
    Reg64              BReg_     = rdx;
    Reg64              AlphaReg_ = rcx;
    Reg64              loopReg_  = rax;
    std::vector<Reg64> elementwiseReg_;

    Label maskLabel_;

    // Note that we remove from the start of this sequence
    // to provide addressing registers for elementwise followed tensors
    // following:
    // https://en.wikipedia.org/wiki/X86_calling_conventions#System_V_AMD64_ABI
    std::vector<Reg64> addressing_registers = {r8,  r9,  r10, r11,
                                               r13, r14, r15, rbx};

    static void print_ld(loop_descriptor const& l)
    {
        LN_LOG(INFO) << "Loop over " << l.var << " from 0 to " << l.end
                     << " by " << l.delta << "\n";
    }

    // This is temporary until I find a better way for nice logging
    // that allows for easy debugging
    std::vector<std::string> tabs = {""};

    std::vector<in_register_tensor_pointer_type> in_register_tensor_pointers;

    using memory_argument = memory_argument_type<vector_size>;

    struct fma_operation
    {
        memory_argument            dest, src1, src2;
        std::map<std::string, int> coordinates;
    };

private:
    // Here we put some default unroll limit.
    static constexpr int default_max_fmas_unrolled = 320;

private:
    std::vector<std::pair<std::string, int>> order;
    std::map<std::string, int> const&        sizes;

    //  allows for arbitrary innermost operations
    std::shared_ptr<operation_pair_base> op_pair;

    std::shared_ptr<elementwise_operation<ISA>> elementwise_preop;
    std::shared_ptr<elementwise_operation<ISA>> elementwise_postop;

    std::set<std::string> const& C_formula;
    std::set<std::string> const& A_formula;
    std::set<std::string> const& B_formula;

    std::map<std::string, int>              C_strides;
    std::map<std::string, int>              A_strides;
    std::map<std::string, int>              B_strides;
    std::vector<std::map<std::string, int>> elementwise_preop_strides;
    std::vector<std::map<std::string, int>> elementwise_postop_strides;
    // concatenation of preop followed by postop strides
    std::vector<std::map<std::string, int>> elementwise_strides;

    int nest_depth;

    // Which can be overriten by the caller.
    int max_fmas_unrolled;

    // Tensors are vectorized if they are looped over in the innermost
    // loop and if the appropriate strides are 1.
    bool is_C_vectorized;
    bool is_A_vectorized;
    bool is_B_vectorized;

    // How do we compute the lements of C.  It's either vectorized,
    // computing vector_size elements at once, or one scalar value at
    // the time, in which case we still use vector instructions, but
    // perform a horizontal sum at the end.
    /* int C_access_len; depricated for C_traits.access_len */

    // Labels holding strides along LSD of vectorized tensors that are
    // not packed.
    Label                               C_access_strides_label;
    Label                               A_access_strides_label;
    Label                               B_access_strides_label;
    std::vector<std::shared_ptr<Label>> elementwise_labels;

    // Tensor traits
    tensor_traits              C_traits;
    tensor_traits              A_traits;
    tensor_traits              B_traits;
    std::vector<tensor_traits> elementwise_traits;

    // stores flags for turning on/off certain loop_nest optimizations
    OptimizationConfiguration optim_config;

    // The name of the variable in the innermost loop (along which the
    // vectorization is performed)
    std::string vectorized_var;

    // Assignment of registers for register blocking of the values of C
    std::map<memory_argument, multi_zmm> C_VMMs;

    // Number of auxillary registers (used for pre-loading and bradcasting, as
    // well as horizontal add at the end)
    int auxiliary_registers;

    // Number of available vector registers for computing
    int available_registers;

    // First register that will not be used by C or auxiliary
    // registers.  Can be used for software pipelining.  Set to
    // isa_traits<ISA>::total_vector_registers if none available
    int first_unused_vmm_register;

    // Some information about the nested loops.  It is kept constant.
    // To be extended with more rich info in the future.
    std::vector<loop_descriptor> loops;

    // Limits per nested partition of the variable. This will be used
    // to figure out loop tails when the loop stride doesn't divide
    // the total loop count.  Heavily used in the recursive loop
    // issuing methods.  The back of the vector represents the current
    // limit in the recursion (nest).
    std::map<std::string, std::vector<int>> limits;

    // Another utility member to be used during the recursive loop
    // visiting methods.  Keeps the current coordinate.
    std::map<std::string, int> current_coordinate_cursor;

private:
    void allocate_elementwise_addressing_registers()
    {
        // TODO(j): for now we assume at most 2 followed
        // tensors for simplicity
        // We allocate the first 2 registers (r8 and r9), which
        // will point to the 5th and 6th arguments in the function
        // generated (as per x86 64 ABI)

        strong_assert(elementwise_strides.size() <= 2);
        for (int i = 0; i < elementwise_strides.size(); i++)
        {
            Reg64 reg = addressing_registers[0];
            elementwiseReg_.push_back(reg);
            addressing_registers.erase(addressing_registers.begin());
        }
    }

    void allocate_elementwise_labels()
    {
        // create labels for each elementwise followed tensor
        // Note that we create for all, but only
        // issue embedded constants later if vectorized and strided
        for (int i = 0; i < elementwise_strides.size(); i++)
        {
            std::shared_ptr<Label> label = std::make_shared<Label>();
            elementwise_labels.push_back(label);
        }
    }

    void initialize_elementwise_ops()
    {
        if (elementwise_preop != nullptr)
        {
            std::vector<tensor_traits> elementwise_preop_traits;
            for (int i = 0; i < elementwise_preop_strides.size(); i++)
            {
                elementwise_preop_traits.push_back(elementwise_traits[i]);
            }
            elementwise_preop->initialize(elementwise_preop_strides,
                                          elementwise_preop_traits,
                                          &maskLabel_);
        }

        if (elementwise_postop != nullptr)
        {
            std::vector<tensor_traits> elementwise_postop_traits;
            int num_preop = elementwise_preop_strides.size();

            for (int i = num_preop;
                 i < num_preop + elementwise_postop_strides.size(); i++)
            {
                elementwise_postop_traits.push_back(elementwise_traits[i]);
            }

            elementwise_postop->initialize(elementwise_postop_strides,
                                           elementwise_postop_traits,
                                           &maskLabel_);
        }
    }

    bool
    is_inside_current_limits(std::map<std::string, int> const& coordinate) const
    {
        bool is_inside = true;
        for (auto const& [var, val] : coordinate)
        {
            is_inside =
                is_inside && limits.count(var) && (val < limits.at(var).back());
        }
        return is_inside;
    }

    int get_cursor_offset(std::map<std::string, int> const& strides)
    {
        int off = 0;
        for (auto const& s : strides)
        {
            off += current_coordinate_cursor[s.first] * s.second;
        }
        return off;
    }

    void collect_loads_and_stores_below_helper(std::set<memory_argument>& ret,
                                               int                        depth)
    {
        auto const& loop             = loops[depth];
        auto        saved_coordinate = current_coordinate_cursor[loop.var];

        if (depth == loops.size() - 1)
        {
            if (C_formula.count(loop.var))
            {
                auto fullIterations = limits[loop.var].back() / vector_size;
                auto rest           = limits[loop.var].back() % vector_size;

                for (int i = 0; i < fullIterations; ++i)
                {
                    ret.insert(memory_argument{get_cursor_offset(C_strides),
                                               &C_traits, vector_size,
                                               current_coordinate_cursor});
                    current_coordinate_cursor[loop.var] += vector_size;
                }

                if (rest)
                {
                    ret.insert(memory_argument{get_cursor_offset(C_strides),
                                               &C_traits, rest,
                                               current_coordinate_cursor});
                }

                current_coordinate_cursor[loop.var] = saved_coordinate;
            }
            else
            {
                ret.insert(memory_argument{get_cursor_offset(C_strides),
                                           &C_traits, vector_size,
                                           current_coordinate_cursor});
            }
        }
        else
        {
            auto num_loop_iterations = limits[loop.var].back() / loop.delta;

            limits[loop.var].push_back(loop.delta);
            for (int i = 0; i < num_loop_iterations; ++i)
            {
                collect_loads_and_stores_below_helper(ret, depth + 1);
                current_coordinate_cursor[loop.var] += loop.delta;
            }
            limits[loop.var].pop_back();

            auto tail = limits[loop.var].back() % loop.delta;
            if (tail)
            {
                limits[loop.var].push_back(tail);
                collect_loads_and_stores_below_helper(ret, depth + 1);
                limits[loop.var].pop_back();
            }

            current_coordinate_cursor[loop.var] = saved_coordinate;
        }
    }

    // Collects all loads and stores below a certain loop in the nest.
    // Assumes that the limits are correctly set for the current loop
    // in the execution tree of the loop nest.  This is to correctly
    // handle the tail cases.
    std::set<memory_argument> collect_loads_and_stores_below(int depth)
    {
        std::set<memory_argument> ret;
        collect_loads_and_stores_below_helper(ret, depth);
        return ret;
    }

    void
    collect_default_loads_and_stores_at_helper(std::set<memory_argument>& ret,
                                               int cur_depth, int req_depth)
    {
        if (cur_depth == req_depth)
        {
            collect_loads_and_stores_below_helper(ret, cur_depth);
        }
        else
        {
            auto const& loop = loops[cur_depth];
            limits[loop.var].push_back(loop.delta);
            collect_default_loads_and_stores_at_helper(ret, cur_depth + 1,
                                                       req_depth);
            limits[loop.var].pop_back();
        }
    }

    // Collect all loads and stores below the first instance of the
    // loop of given depth in the execution tree.  This is basically
    // all the loads and stores below depth, where all the parent
    // loops are not in their tail.
    std::set<memory_argument> collect_default_loads_and_stores_at(int depth)
    {
        std::set<memory_argument> ret;
        collect_default_loads_and_stores_at_helper(ret, 0, depth);
        return ret;
    }

    void collect_unrolled_FMAs_below_helper(std::vector<fma_operation>& ret,
                                            int                         depth)
    {
        auto const& loop             = loops[depth];
        auto        saved_coordinate = current_coordinate_cursor[loop.var];

        if (depth == nest_depth - 1) // last, vectorized loop
        {
            strong_assert(loop.delta == 1);

            auto fullIterations = limits[loop.var].back() / vector_size;
            auto rest           = limits[loop.var].back() % vector_size;

            for (int i = 0; i < fullIterations; ++i)
            {
                memory_argument dest{get_cursor_offset(C_strides), &C_traits,
                                     vector_size};
                memory_argument src1{get_cursor_offset(B_strides), &B_traits,
                                     vector_size};
                memory_argument src2{get_cursor_offset(A_strides), &A_traits,
                                     vector_size};

                ret.push_back({dest, src1, src2, current_coordinate_cursor});
                current_coordinate_cursor[loop.var] += vector_size;
            }

            if (rest)
            {
                memory_argument dest{get_cursor_offset(C_strides), &C_traits,
                                     rest};
                memory_argument src1{get_cursor_offset(B_strides), &B_traits,
                                     rest};
                memory_argument src2{get_cursor_offset(A_strides), &A_traits,
                                     rest};

                ret.push_back({dest, src1, src2, current_coordinate_cursor});
            }

            current_coordinate_cursor[loop.var] = saved_coordinate;
        }
        else
        {
            for (int i = 0; i < limits[loop.var].back() / loop.delta; ++i)
            {
                limits[loop.var].push_back(loop.delta);
                collect_unrolled_FMAs_below_helper(ret, depth + 1);
                limits[loop.var].pop_back();
                current_coordinate_cursor[loop.var] += loop.delta;
            }

            auto tail = limits[loop.var].back() % loop.delta;

            if (tail)
            {
                limits[loop.var].push_back(tail);
                collect_unrolled_FMAs_below_helper(ret, depth + 1);
                limits[loop.var].pop_back();
            }

            current_coordinate_cursor[loop.var] = saved_coordinate;
        }
    }

    // Collects all (unrolled) FMAs below a certain loop in the nest.
    // Assumes that the limits are correctly set for the current loop
    // in the execution tree of the loop nest.  This is to correctly
    // handle the tail cases.
    std::vector<fma_operation> collect_unrolled_FMAs_below(int depth)
    {
        std::vector<fma_operation> ret;
        collect_unrolled_FMAs_below_helper(ret, depth);
        return ret;
    }

    void
    collect_default_unrolled_FMAs_at_helper(std::vector<fma_operation>& ret,
                                            int cur_depth, int req_depth)
    {
        if (cur_depth == req_depth)
        {
            collect_unrolled_FMAs_below_helper(ret, cur_depth);
        }
        else
        {
            auto const& loop = loops[cur_depth];
            limits[loop.var].push_back(loop.delta);
            collect_default_unrolled_FMAs_at_helper(ret, cur_depth + 1,
                                                    req_depth);
            limits[loop.var].pop_back();
        }
    }

    // Collect all (unrolled) FMAs below the first instance of the
    // loop of given depth in the execution tree.  Each other instance
    // of the loop in the tree at the same depth will contain a subset
    // of the collected FMAs as it will be in a tail of at least one
    // loop.
    std::vector<fma_operation> collect_default_unrolled_FMAs_at(int depth)
    {
        std::vector<fma_operation> ret;
        collect_default_unrolled_FMAs_at_helper(ret, 0, depth);
        return ret;
    }

    // Pushes the "followed" pointers (C, A or B, and any extra ons
    // that will be used by the future arbitrary innermost operations)
    // that have strides along the dimension dim.
    void push_pointers(std::string const& dim)
    {
        for (auto const& ptr : in_register_tensor_pointers)
        {
            if (ptr.strides.count(dim) && ptr.strides.at(dim) != 0)
            {
                LN_LOG(INFO) << tabs.back() << "PUSH " << ptr.name << "("
                             << ptr.reg.toString() << ")\n";
                push(ptr.reg);
            }
        }
    }

    // Similarly pops the pointers
    void pop_pointers(std::string const& dim)
    {
        for (auto it = in_register_tensor_pointers.rbegin();
             it != in_register_tensor_pointers.rend(); ++it)
        {
            auto const& ptr = *it;
            if (ptr.strides.count(dim) && ptr.strides.at(dim) != 0)
            {
                LN_LOG(INFO) << tabs.back() << "POP " << ptr.name << "("
                             << ptr.reg.toString() << ")\n";
                pop(ptr.reg);
            }
        }
    };

    // Similarly advances the pointers by delta elements along the
    // given dimension
    void advance_pointers(std::string const& dim, int delta)
    {
        for (auto const& ptr : in_register_tensor_pointers)
        {
            if (ptr.strides.count(dim) && ptr.strides.at(dim) != 0)
            {
                LN_LOG(INFO)
                    << tabs.back() << ptr.name << "(" << ptr.reg.toString()
                    << ") += " << delta << " * " << ptr.strides.at(dim) << "\n";
                add(ptr.reg, ptr.strides.at(dim) * delta * 4);
            }
        }
    };

    void scatter_avx2_register(Ymm ymm, int mask, Xbyak::RegExp const& base,
                               int stride)
    {
        // TODO(zi) decide whether to keep both methods, each with its
        // own trade-offs.

        vmovups(ptr[rsp - 32], ymm);
        for (int i = 0; i < mask; ++i)
        {
            mov(r12.cvt32(), dword[rsp - 32 + i * 4]);
            mov(dword[base + i * stride], r12.cvt32());
        }

        // The commented out method below has more instructions (and
        // much more instruction bytes) but requires fewer memory
        // accesses.  It probably doens't matter, as the reads in the
        // approach above can be pipelined with the writes.  Need a
        // good benchmark, or maybe a binary parameter?

        // if (mask == 0)
        //     return;

        // vmovups(ymm1, ymm);
        // vmovss(ptr[base], xmm1);

        // if (mask == 1)
        //     return;

        // vpermilps(xmm0, xmm1, 177);
        // vmovss(ptr[base + stride], xmm0);

        // if (mask == 2)
        //     return;

        // vpermilpd(xmm0, xmm1, 1);
        // vmovss(ptr[base + stride * 2], xmm0);

        // if (mask == 3)
        //     return;

        // vpermilps(xmm1, xmm0, 177);
        // vmovss(ptr[base + stride * 3], xmm1);

        // if (mask == 4)
        //     return;

        // vmovups(ymm1, ymm);
        // vextractf128(xmm0, ymm1, 1);
        // vmovss(ptr[base + stride * 4], xmm0);

        // if (mask == 5)
        //     return;

        // vpermilps(xmm1, xmm0, 177);
        // vmovss(ptr[base + stride * 5], xmm1);

        // if (mask == 6)
        //     return;

        // vpermilpd(xmm1, xmm0, 1);
        // vmovss(ptr[base + stride * 6], xmm1);

        // if (mask == 7)
        //     return;

        // vpermilps(xmm0, xmm1, 177);
        // vmovss(ptr[base + stride * 7], xmm0);
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx512> || std::is_same_v<R, avx2_plus>>
    issue_C_loads(std::set<memory_argument> const& loads,
                  std::optional<int>               tail_mask)
    {
        Vmm    arg_C_strides;
        int    next_vector_register = 0;
        OpMask tail_k_mask          = k2;
        OpMask full_k_mask          = k3;
        OpMask temp_k_mask          = k4;

        if (C_traits.access == VECTOR_STRIDED)
        {
            arg_C_strides = Vmm(next_vector_register++);
            vmovups(arg_C_strides, ptr[rip + C_access_strides_label]);
            mov(r12, (1 << vector_size) - 1);
            kmovw(full_k_mask, r12.cvt32());
        }

        if (tail_mask)
        {
            mov(r12, (1 << (*tail_mask)) - 1);
            kmovw(tail_k_mask, r12.cvt32());

            strong_assert(C_traits.access != SCALAR);
        }

        strong_assert(next_vector_register <= auxiliary_registers);

        for (auto const& c : loads)
        {
            LN_LOG(INFO) << tabs.back() << "LOAD " << c.readable() << "\n";

            switch (C_traits.access)
            {
            case SCALAR:
                op_pair->set_to_identity(*this, C_VMMs[c][0]);
                vmovss(Xmm(C_VMMs[c][0].getIdx()), ptr[CReg_ + c.offset * 4]);
                break;

            case VECTOR_PACKED:
                if (c.mask == vector_size) // no mask
                    vmovups(C_VMMs[c][0], ptr[CReg_ + c.offset * 4]);
                else
                    vmovups(C_VMMs[c][0] | tail_k_mask,
                            ptr[CReg_ + c.offset * 4]);
                break;

            case VECTOR_STRIDED:
                kmovw(temp_k_mask, // The mask gets updated in gather
                      (c.mask == vector_size ? full_k_mask : tail_k_mask));
                vgatherdps(C_VMMs[c][0] | temp_k_mask,
                           ptr[CReg_ + c.offset * 4 + arg_C_strides]);
                break;
            }

            // Set auxiliary horizontal vector regs to identity value
            for (int s = 1; s < C_VMMs[c].size(); ++s)
            {
                op_pair->set_to_identity(*this, C_VMMs[c][s]);
            }
        }
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx2>>
    issue_C_loads(std::set<memory_argument> const& loads,
                  std::optional<int>               tail_mask)
    {
        Ymm arg_C_strides;
        Ymm ymm_full_mask, ymm_tail_mask, ymm_temp_mask;
        int next_vector_register = 0;

        if (C_traits.access == VECTOR_STRIDED)
        {
            arg_C_strides = Ymm(next_vector_register++);
            vmovups(arg_C_strides, ptr[rip + C_access_strides_label]);

            ymm_full_mask = Vmm(next_vector_register++);
            vmovups(ymm_full_mask, ptr[rip + maskLabel_]);

            ymm_temp_mask = Vmm(next_vector_register++);
        }

        if (tail_mask)
        {
            ymm_tail_mask = Vmm(next_vector_register++);
            vmovups(ymm_tail_mask,
                    ptr[rip + maskLabel_ + 4 * (8 - (*tail_mask))]);
        }

        strong_assert(next_vector_register <= auxiliary_registers);

        // issue loads
        for (auto const& c : loads)
        {
            LN_LOG(INFO) << tabs.back() << "LOAD " << c.readable() << "\n";

            switch (C_traits.access)
            {
            case SCALAR:
                op_pair->set_to_identity(*this, C_VMMs[c][0]);
                vmovss(Xmm(C_VMMs[c][0].getIdx()), ptr[CReg_ + c.offset * 4]);
                break;

            case VECTOR_PACKED:
                if (c.mask == vector_size)
                    vmovups(C_VMMs[c][0], ptr[CReg_ + c.offset * 4]);
                else
                    vmaskmovps(C_VMMs[c][0], ymm_tail_mask,
                               ptr[CReg_ + c.offset * 4]);
                break;

            case VECTOR_STRIDED:
                vmovups(ymm_temp_mask, (c.mask == vector_size ? ymm_full_mask
                                                              : ymm_tail_mask));
                vgatherdps(C_VMMs[c][0],
                           ptr[CReg_ + c.offset * 4 + arg_C_strides],
                           ymm_temp_mask);
                break;
            }

            // Set auxiliary horizontal vector regs to identity value
            for (int s = 1; s < C_VMMs[c].size(); ++s)
            {
                op_pair->set_to_identity(*this, C_VMMs[c][s]);
            }
        }
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx512> || std::is_same_v<R, avx2_plus>>
    issue_C_elementwise_preop(std::set<memory_argument> const& loads,
                              std::optional<int>               tail_mask)
    {
        OpMask tail_k_mask = k2;
        OpMask full_k_mask = k3;
        OpMask temp_k_mask = k4;

        switch (C_traits.access)
        {
        case VECTOR_STRIDED:
            /* fall through to vector packed case */
        case VECTOR_PACKED:
        {
            std::vector<Vmm> auxillary = {Vmm(0), Vmm(1)};

            if (tail_mask)
            {
                mov(r12, (1 << (*tail_mask)) - 1);
                kmovw(tail_k_mask, r12.cvt32());

                strong_assert(C_traits.access_len != 1);
            }

            std::vector<std::pair<memory_argument, Vmm>> loads_and_regs;
            for (auto const& c : loads)
            {
                loads_and_regs.push_back({c, C_VMMs[c][0]});
            }

            elementwise_preop->process_batch(
                *this, loads_and_regs, auxillary, {full_k_mask, temp_k_mask},
                C_traits.access, R(),
                tail_mask ? std::optional<OpMask>(tail_k_mask) : std::nullopt);
        }
        break;

        case SCALAR:
        {
            std::vector<Xmm> auxillary = {xmm0, xmm1};

            std::vector<std::pair<memory_argument, Xmm>> loads_and_regs;
            for (auto const& c : loads)
            {
                loads_and_regs.push_back({c, Xmm(C_VMMs[c][0].getIdx())});
            }

            elementwise_preop->process_batch(
                *this, loads_and_regs, {xmm0, xmm1}, C_traits.access, R());
        }
        break;
        }
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx2>>
    issue_C_elementwise_preop(std::set<memory_argument> const& loads,
                              std::optional<int>               tail_mask)
    {
        switch (C_traits.access)
        {
        case VECTOR_STRIDED:
            /* fall through to vector packed case */
        case VECTOR_PACKED:
        {
            // Vmm0 and Vmm1 are kept free by convention as auxillary
            std::vector<Vmm> auxillary = {Vmm(0), Vmm(1)};
            Ymm              ymm_tail_mask;

            // may have others available too
            for (int i = first_unused_vmm_register;
                 i < isa_traits<ISA>::total_vector_registers; i++)
            {
                auxillary.push_back(Vmm(i));
            }

            if (tail_mask)
            {
                ymm_tail_mask = auxillary.back();
                vmovups(ymm_tail_mask,
                        ptr[rip + maskLabel_ + 4 * (8 - (*tail_mask))]);
                auxillary.pop_back();
            }

            std::vector<std::pair<memory_argument, Vmm>> loads_and_regs;
            for (auto const& c : loads)
            {
                loads_and_regs.push_back({c, C_VMMs[c][0]});
            }

            elementwise_preop->process_batch(
                *this, loads_and_regs, auxillary, C_traits.access, R(),
                tail_mask ? std::optional<Ymm>(ymm_tail_mask) : std::nullopt);
        }
        break;

        case SCALAR:
        {
            std::vector<Xmm> auxillary = {xmm0, xmm1};

            std::vector<std::pair<memory_argument, Xmm>> loads_and_regs;
            for (auto const& c : loads)
            {
                loads_and_regs.push_back({c, Xmm(C_VMMs[c][0].getIdx())});
            }

            elementwise_preop->process_batch(
                *this, loads_and_regs, {xmm0, xmm1}, C_traits.access, R());
        }
        break;
        }
    }

    void issue_C_loads(std::set<memory_argument> const& loads,
                       bool                             issue_first_alpha_logic)
    {

        for (auto& CVmm : C_VMMs)
        {
            CVmm.second.reset();
        }

        // TODO (relax)
        std::optional<int> tail_mask;

        for (auto const& c : loads)
        {
            if (c.mask != vector_size)
            {
                strong_assert(!tail_mask || *tail_mask == c.mask);
                tail_mask = c.mask;
            }
        }

        if (issue_first_alpha_logic)
        {
            Label loadDataLabel;
            Label doneInitLabel;

            cmp(AlphaReg_, 0);
            jg(loadDataLabel, T_NEAR);

            // Same code among all ISAs for initializing registers to
            // zero/identity
            for (auto const& c : loads)
            {
                LN_LOG(INFO)
                    << tabs.back() << "ZERO(identity) " << c.readable() << "\n";
                for (int s = 0; s < C_VMMs[c].size(); ++s)
                {
                    op_pair->set_to_identity(*this, C_VMMs[c][s]);
                }
            }

            jmp(doneInitLabel, T_NEAR);

            L(loadDataLabel);

            issue_C_loads(loads, tail_mask);

            if (elementwise_preop != nullptr)
            {
                Label donePreOp;

                cmp(AlphaReg_, 1);
                jne(donePreOp, T_NEAR);

                for (auto const& c : loads)
                {
                    LN_LOG(INFO)
                        << tabs.back() << elementwise_preop->name() << " "
                        << c.readable() << " at (" << 0 << ")\n";
                }

                issue_C_elementwise_preop(loads, tail_mask);
                L(donePreOp);
            }

            L(doneInitLabel);
        }
        else
        {
            issue_C_loads(loads, tail_mask);
        }
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx512> || std::is_same_v<R, avx2_plus>>
    issue_C_stores(std::set<memory_argument> const& stores,
                   std::optional<int> tail_mask, int max_alpha,
                   bool issue_max_alpha_logic)
    {
        Vmm    arg_C_strides;
        int    next_vector_register = 0;
        OpMask tail_k_mask          = k2;
        OpMask full_k_mask          = k3;
        OpMask temp_k_mask          = k4;

        if (issue_max_alpha_logic && elementwise_postop)
        {
            if (C_traits.access == VECTOR_PACKED ||
                C_traits.access == VECTOR_STRIDED)
            {
                Label not_last_label;
                cmp(AlphaReg_, max_alpha - 1);
                jl(not_last_label, T_NEAR);

                // // TODO(zi): Better use of auxiliary array and extra classes
                std::vector<std::pair<memory_argument, Xbyak::Zmm>>
                    stores_and_regs;

                for (auto const& c : stores)
                {
                    C_VMMs[c].reduce(*this, op_pair);
                    stores_and_regs.push_back({c, C_VMMs[c][0]});

                    LN_LOG(INFO)
                        << tabs.back() << elementwise_postop->name() << " "
                        << c.readable() << " at (" << max_alpha << ")\n";
                }

                if (tail_mask)
                {
                    mov(r12, (1 << (*tail_mask)) - 1);
                    kmovw(tail_k_mask, r12.cvt32());

                    strong_assert(C_traits.access_len != 1);
                }

                elementwise_postop->process_batch(
                    *this, stores_and_regs, {Vmm(0), Vmm(1)},
                    {full_k_mask, temp_k_mask}, C_traits.access, R(),
                    tail_mask ? std::optional<OpMask>(tail_k_mask)
                              : std::nullopt);

                L(not_last_label);
            }
        }

        if (C_traits.access == VECTOR_STRIDED)
        {
            arg_C_strides = Vmm(next_vector_register++);
            vmovups(arg_C_strides, ptr[rip + C_access_strides_label]);
            mov(r12, (1 << vector_size) - 1); // TODO (this is probably already
            kmovw(full_k_mask, r12.cvt32());  // initialized during
            // loads)? Add logic to
            // check that, and skip
            // if not necessary
        }

        strong_assert(next_vector_register <= auxiliary_registers);

        if (tail_mask)
        {
            mov(r12, (1 << (*tail_mask)) - 1);
            kmovw(tail_k_mask, r12.cvt32());

            strong_assert(C_traits.access_len != 1);
        }

        for (auto const& c : stores)
        {
            LN_LOG(INFO) << tabs.back() << "STORE " << c.readable() << "\n";

            C_VMMs[c].reduce(*this, op_pair);
            Label not_last_label;

            switch (C_traits.access)
            {
            case SCALAR:
                if constexpr (std::is_same_v<ISA, avx512>)
                {
                    // Needs the horizontal sum
                    vextractf64x4(ymm1, C_VMMs[c][0], 1);
                    if (C_VMMs[c][0].getIdx() < 16)
                    {
                        op_pair->plus(*this, ymm1, ymm1,
                                      Ymm(C_VMMs[c][0].getIdx()));
                    }
                    else
                    {
                        op_pair->plus(*this, zmm1, zmm1, C_VMMs[c][0]);
                    }

                    vextractf128(xmm0, ymm1, 1);
                    op_pair->plus(*this, xmm0, xmm0, xmm1);
                }
                else // avx2_plus
                {
                    vextractf128(xmm0, C_VMMs[c][0], 1);
                    op_pair->plus(*this, xmm0, xmm0, C_VMMs[c][0]);
                }

                // xmm1 = xmm0[1,0]
                vpermilpd(xmm1, xmm0, 1);
                op_pair->plus(*this, xmm0, xmm0, xmm1);

                // xmm1 = xmm0[1,0,3,2]
                vpermilps(xmm1, xmm0, 177);
                op_pair->plus(*this, xmm0, xmm0, xmm1);

                if (issue_max_alpha_logic && elementwise_postop)
                {
                    cmp(AlphaReg_, max_alpha - 1);
                    jl(not_last_label);

                    LN_LOG(INFO)
                        << tabs.back() << elementwise_postop->name() << " "
                        << c.readable() << " at (" << max_alpha << ")\n";

                    elementwise_postop->process_batch(
                        *this, {{c, xmm0}}, {xmm1}, C_traits.access, R());
                    L(not_last_label);
                }

                vmovss(ptr[CReg_ + c.offset * 4], xmm0);
                break;

            case VECTOR_PACKED:
                if (c.mask == vector_size)
                    vmovups(ptr[CReg_ + c.offset * 4], C_VMMs[c][0]);
                else
                    vmovups(ptr[CReg_ + c.offset * 4] | tail_k_mask,
                            C_VMMs[c][0]);
                break;

            case VECTOR_STRIDED:
                kmovw(temp_k_mask, // The mask gets updated in gather
                      (c.mask == vector_size ? full_k_mask : tail_k_mask));
                vscatterdps(ptr[CReg_ + c.offset * 4 + arg_C_strides] |
                                temp_k_mask,
                            C_VMMs[c][0]);
                break;
            }
        }
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx2>>
    issue_C_stores(std::set<memory_argument> const& stores,
                   std::optional<int> tail_mask, int max_alpha,
                   bool issue_max_alpha_logic)
    {
        Ymm arg_C_strides;
        Ymm ymm_tail_mask;
        int next_vector_register = 0;

        if (issue_max_alpha_logic && elementwise_postop)
        {
            if (C_traits.access == VECTOR_PACKED ||
                C_traits.access == VECTOR_STRIDED)
            {
                Label not_last_label;
                cmp(AlphaReg_, max_alpha - 1);
                jl(not_last_label, T_NEAR);

                std::vector<std::pair<memory_argument, Xbyak::Ymm>>
                    stores_and_regs;

                for (auto const& c : stores)
                {
                    C_VMMs[c].reduce(*this, op_pair);
                    stores_and_regs.push_back({c, C_VMMs[c][0]});

                    LN_LOG(INFO)
                        << tabs.back() << elementwise_postop->name() << " "
                        << c.readable() << " at (" << max_alpha << ")\n";
                }

                // Ymm0 and Ymm1 are kept free by convention as auxillary
                std::vector<Vmm> auxillary = {Vmm(0), Vmm(1)};
                // may have others available too
                for (int i = first_unused_vmm_register;
                     i < isa_traits<ISA>::total_vector_registers; i++)
                {
                    auxillary.push_back(Vmm(i));
                }

                if (tail_mask)
                {
                    ymm_tail_mask = auxillary.back();
                    vmovups(ymm_tail_mask,
                            ptr[rip + maskLabel_ + 4 * (8 - (*tail_mask))]);
                    auxillary.pop_back();
                }

                elementwise_postop->process_batch(
                    *this, stores_and_regs, auxillary, C_traits.access, R(),
                    tail_mask ? std::optional<Vmm>(ymm_tail_mask)
                              : std::nullopt);

                L(not_last_label);
            }
        }

        if (tail_mask)
        {
            ymm_tail_mask = Vmm(next_vector_register++);
            vmovups(ymm_tail_mask,
                    ptr[rip + maskLabel_ + 4 * (8 - (*tail_mask))]);
        }

        strong_assert(next_vector_register <= auxiliary_registers);

        for (auto const& c : stores)
        {
            LN_LOG(INFO) << tabs.back() << "STORE " << c.readable() << "\n";

            C_VMMs[c].reduce(*this, op_pair);

            Label not_last_label;

            switch (C_traits.access)
            {
            case SCALAR:
                vextractf128(xmm0, C_VMMs[c][0], 1);
                op_pair->plus(*this, xmm0, xmm0, C_VMMs[c][0]);

                // xmm1 = xmm0[1,0]
                vpermilpd(xmm1, xmm0, 1);
                op_pair->plus(*this, xmm0, xmm0, xmm1);

                // xmm1 = xmm0[1,0,3,2]
                vpermilps(xmm1, xmm0, 177);
                op_pair->plus(*this, xmm0, xmm0, xmm1);

                if (issue_max_alpha_logic && elementwise_postop)
                {
                    cmp(AlphaReg_, max_alpha - 1);
                    jl(not_last_label);

                    LN_LOG(INFO)
                        << tabs.back() << elementwise_postop->name() << " "
                        << c.readable() << " at (" << max_alpha << ")\n";

                    elementwise_postop->process_batch(
                        *this, {{c, xmm0}}, {xmm1}, C_traits.access, R());
                    L(not_last_label);
                }

                vmovss(ptr[CReg_ + c.offset * 4], xmm0);
                break;

            case VECTOR_PACKED:
                if (c.mask == vector_size)
                    vmovups(ptr[CReg_ + c.offset * 4], C_VMMs[c][0]);
                else
                    vmaskmovps(ptr[CReg_ + c.offset * 4], ymm_tail_mask,
                               C_VMMs[c][0]);
                break;

            case VECTOR_STRIDED:
                scatter_avx2_register(
                    C_VMMs[c][0],
                    c.mask == vector_size ? vector_size : *tail_mask,
                    CReg_ + c.offset * 4, C_strides.at(vectorized_var) * 4);
                break;
            }
        }
    }

    void issue_C_stores(std::set<memory_argument> const& stores, int max_alpha,
                        bool issue_max_alpha_logic)
    {
        std::optional<int> tail_mask;

        for (auto const& c : stores)
        {
            if (c.mask != vector_size)
            {
                strong_assert(!tail_mask || *tail_mask == c.mask);
                tail_mask = c.mask;
            }
        }

        issue_C_stores(stores, tail_mask, max_alpha, issue_max_alpha_logic);
    }

    void issue_unrolled_fmas_scalar_vector()
    {

        auto instructions = std::move(instruction_IRs.front());
        strong_assert(instruction_IRs.size());
        instruction_IRs.pop_front();

        for (auto const& insn : instructions)
        {
            std::visit(
                overloaded{
                    [&](load_instruction const& i) {
                        if (i.does_broadcast)
                        {
                            vbroadcastss(Vmm(i.vmm_idx),
                                         ptr[Reg64(i.tensor_loc.idx) +
                                             i.tensor_loc.offset]);
                        }
                        else
                        {
                            vmovups(Vmm(i.vmm_idx),
                                    ptr[Reg64(i.tensor_loc.idx) +
                                        i.tensor_loc.offset]);
                        }
                    },
                    [&](fmla_instruction const& fml) {
                        if (std::holds_alternative<int>(fml.right_src))
                        {
                            int rreg = std::get<int>(fml.right_src);
                            if (op_pair->can_fuse())
                            {
                                op_pair->fuse(*this, Vmm(fml.dst),
                                              Vmm(fml.left_src), Vmm(rreg));
                            }
                            else
                            {
                                op_pair->multiplies(*this, Vmm(0),
                                                    Vmm(fml.left_src),
                                                    Vmm(rreg));
                                op_pair->plus(*this, Vmm(fml.dst), Vmm(fml.dst),
                                              Vmm(0));
                            }
                        }
                        else
                        {
                            auto loc = std::get<memory_src>(fml.right_src);

                            if (loc.broadcast)
                            {
                                if (op_pair->can_fuse())
                                {
                                    op_pair->fuse(
                                        *this, Vmm(fml.dst), Vmm(fml.left_src),
                                        ptr_b[Reg64(loc.idx) + loc.offset]);
                                }
                                else
                                {
                                    op_pair->multiplies(
                                        *this, Vmm(0), Vmm(fml.left_src),
                                        ptr_b[Reg64(loc.idx) + loc.offset]);
                                    op_pair->plus(*this, Vmm(fml.dst),
                                                  Vmm(fml.dst), Vmm(0));
                                }
                            }
                            else
                            {
                                if (op_pair->can_fuse())
                                {
                                    op_pair->fuse(
                                        *this, Vmm(fml.dst), Vmm(fml.left_src),
                                        ptr[Reg64(loc.idx) + loc.offset]);
                                }
                                else
                                {
                                    op_pair->multiplies(
                                        *this, Vmm(0), Vmm(fml.left_src),
                                        ptr[Reg64(loc.idx) + loc.offset]);
                                    op_pair->plus(*this, Vmm(fml.dst),
                                                  Vmm(fml.dst), Vmm(0));
                                }
                            }
                        }
                    }},
                insn);
        }

        // for (auto const& insn : instructions)
        // {
        //     std::visit(
        //         overloaded{[&](load_instruction const& i) {
        //                        int ptr_reg_idx =
        //                        i.tensor_location.tensor_idx; int ptr_offset =
        //                        i.tensor_location.tensor_offset;

        //                        LN_LOG(INFO)
        //                            << tabs.back() << "::LOAD Vreg(" << i.vreg
        //                            << ")[" << i.num_lanes << "], X_"
        //                            << ptr_reg_idx << "[" << ptr_offset <<
        //                            "]\n";
        //                    },
        //                    [&](fmla_instruction const& fml) {
        //                        LN_LOG(INFO) << tabs.back() << "::FMLA Vreg("
        //                                     << fml.dst.number << "), Vreg("
        //                                     << fml.left_src.number << "),
        //                                     Vreg("
        //                                     << fml.right_src.number << ") ["
        //                                     << fml.right_src.lane << "]\n";
        //                    },
        //                    [](std::monostate) {}},
        //         insn);
        // }

        // for (auto const& offs : tensor_offsets)
        // {
        //     sadd_imm(XReg(offs.first), -offs.second);
        // }
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx512> || std::is_same_v<R, avx2_plus>>
    issue_unrolled_fmas(
        std::vector<fma_operation>                            fmas,
        std::map<int, std::shared_ptr<address_packer>> const& addressers,
        std::optional<int>                                    tail_mask)

    {

        if ((A_traits.access == SCALAR && B_traits.access == VECTOR_PACKED) ||
            (A_traits.access == VECTOR_PACKED && B_traits.access == SCALAR) ||
            (A_traits.access == VECTOR_PACKED &&
             B_traits.access == VECTOR_PACKED && C_traits.access != SCALAR))
        {
            issue_unrolled_fmas_scalar_vector();
            return;
        }

        // Assignment of vector registers
        Vmm arg1_register, arg2_register;
        Vmm arg_A_strides, arg_B_strides; // TODO (if same, use same register)

        OpMask tail_k_mask = k2;
        OpMask full_k_mask = k3;
        OpMask temp_k_mask = k4;

        // keep Vmm0 available if not fused, to chain ops
        int next_vector_register = op_pair->can_fuse() ? 0 : 1;

        arg1_register = Vmm(next_vector_register++);
        arg2_register = Vmm(next_vector_register++);

        std::map<std::string, Vmm> tensor_strides;

        if (A_traits.access == VECTOR_STRIDED)
        {
            tensor_strides["A"] = arg_A_strides = Vmm(next_vector_register++);
            vmovups(arg_A_strides, ptr[rip + A_access_strides_label]);
        }

        if (B_traits.access == VECTOR_STRIDED)
        {
            tensor_strides["B"] = arg_B_strides = Vmm(next_vector_register++);
            vmovups(arg_B_strides, ptr[rip + B_access_strides_label]);
        }

        strong_assert(next_vector_register <= auxiliary_registers);

        if (A_traits.access == VECTOR_STRIDED ||
            B_traits.access == VECTOR_STRIDED)
        {
            mov(r12, (1 << vector_size) - 1);
            kmovw(full_k_mask, r12.cvt32());
        }

        int mask_size = -1;

        // TODO(zi) issue mask only when required (for avx512 as well)
        auto const ensure_initalized_mask = [&](int mask) {
            strong_assert(tail_mask && "Tail mask was not detected");
            strong_assert(mask == *tail_mask);

            if (mask_size == -1)
            {
                mask_size = *tail_mask;
                mov(r12, (1 << (*tail_mask)) - 1);
                kmovw(tail_k_mask, r12.cvt32());
            }
        };

        most_frequent_queue<memory_argument> queue;

        for (auto const& p : addressers)
        {
            p.second->loop_prologue();
        }

        for (auto const& inst : fmas)
        {
            // Ensures no instructions are added to the unrolled
            // loop tails
            strong_assert(is_inside_current_limits(inst.coordinates));
            queue.inc(inst.src1);
            queue.inc(inst.src2);
        }

        std::vector<Vmm> arg1_registers;
        arg1_registers.push_back(arg1_register);
        for (int i = first_unused_vmm_register;
             i < isa_traits<ISA>::total_vector_registers; ++i)
        {
            arg1_registers.push_back(Vmm(i));
        }

        // TODO(zi) replace this eyeballed value
        if (arg1_registers.size() > 5)
        {
            arg1_registers.resize(5);
        }

        int cycle = optim_config.delay_innermost_operations()
                        ? arg1_registers.size()
                        : 1;
        int current = 0;

        std::vector<std::function<void()>> issue_delayed_ops(cycle, []() {});

        for (; queue.size() > 0; ++current)
        {
            issue_delayed_ops[current % cycle]();

            auto arg1_reg = arg1_registers[current % cycle];

            auto addr = queue.top();
            queue.pop();

            LN_LOG(INFO) << tabs.back() << "LOAD " << addr.readable() << " ["
                         << addr.mask << "]\n";

            switch (addr.traits->access)
            {
            case SCALAR:
                if (C_traits.access == SCALAR && addr.mask != vector_size)
                {
                    ensure_initalized_mask(addr.mask);
                    vbroadcastss(arg1_reg | tail_k_mask | T_z,
                                 ptr[addressers.at(addr.traits->reg.getIdx())
                                         ->get_address(addr.offset * 4)]);
                }
                else
                {
                    vbroadcastss(arg1_reg,
                                 ptr[addressers.at(addr.traits->reg.getIdx())
                                         ->get_address(addr.offset * 4)]);
                }
                break;

            case VECTOR_PACKED:
                if (C_traits.access == SCALAR && addr.mask != vector_size)
                {
                    ensure_initalized_mask(addr.mask);
                    vmovups(arg1_reg | tail_k_mask | T_z,
                            ptr[addressers.at(addr.traits->reg.getIdx())
                                    ->get_address(addr.offset * 4)]);
                }
                else
                {
                    vmovups(arg1_reg,
                            ptr[addressers.at(addr.traits->reg.getIdx())
                                    ->get_address(addr.offset * 4)]);
                }
                break;

            case VECTOR_STRIDED:
                if (addr.mask == vector_size)
                {
                    kmovw(temp_k_mask, full_k_mask);
                }
                else
                {
                    ensure_initalized_mask(addr.mask);
                    kmovw(temp_k_mask, tail_k_mask);
                    if (C_traits.access == SCALAR)
                    {
                        // Need to zero/identity out the register
                        // as gather doesn't leaves
                        // previous values
                        op_pair->set_to_identity(*this, arg1_reg);
                    }
                }

                vgatherdps(
                    arg1_reg | temp_k_mask,
                    ptr[addressers.at(addr.traits->reg.getIdx())
                            ->get_address_without_index(addr.offset * 4) +
                        tensor_strides[addr.traits->name]]);

                break;
            }

            std::vector<fma_operation> delayed_fma_operations;

            for (auto it = fmas.begin(); it != fmas.end();)
            {
                if (it->src1 == addr || it->src2 == addr)
                {
                    auto src1 = it->src1;
                    auto src2 = it->src2;
                    if (addr == it->src2)
                    {
                        std::swap(src1, src2);
                    }

                    queue.dec(src2);

                    delayed_fma_operations.push_back(*it);

                    LN_LOG(INFO) << tabs.back() << it->dest.readable()
                                 << " += " << it->src1.readable() << " * "
                                 << it->src2.readable() << "\n";
                    it = fmas.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            issue_delayed_ops[current % cycle] = [arg1_reg, addr,
                                                  delayed_fma_operations,
                                                  &addressers, temp_k_mask,
                                                  full_k_mask, tail_k_mask,
                                                  this, &ensure_initalized_mask,
                                                  arg2_register,
                                                  &tensor_strides]() {
                bool first = true;
                for (auto const& op : delayed_fma_operations)
                {
                    auto src1 = op.src1;
                    auto src2 = op.src2;

                    if (addr == src2)
                    {
                        std::swap(src1, src2);
                    }

                    if (first)
                    {
                        first = false;
                        addressers.at(src2.traits->reg.getIdx())
                            ->move_to(src2.offset * 4);
                    }

                    switch (src2.traits->access)
                    {
                    case SCALAR:
                        if (op_pair->can_fuse())
                        {
                            op_pair->fuse(
                                *this, C_VMMs[op.dest]++, arg1_reg,
                                ptr_b[addressers.at(src2.traits->reg.getIdx())
                                          ->get_address(src2.offset * 4)]);
                        }
                        else
                        {
                            Vmm auxiliary = Vmm(0);
                            Vmm dest      = C_VMMs[op.dest]++;

                            op_pair->multiplies(
                                *this, auxiliary, arg1_reg,
                                ptr_b[addressers.at(src2.traits->reg.getIdx())
                                          ->get_address(src2.offset * 4)]);
                            op_pair->plus(*this, dest, dest, auxiliary);
                        }
                        break;

                    case VECTOR_PACKED:
                        if (op_pair->can_fuse())
                        {
                            op_pair->fuse(
                                *this, C_VMMs[op.dest]++, arg1_reg,
                                ptr[addressers.at(src2.traits->reg.getIdx())
                                        ->get_address(src2.offset * 4)]);
                        }
                        else
                        {
                            Vmm auxiliary = Vmm(0);
                            Vmm dest      = C_VMMs[op.dest]++;

                            op_pair->multiplies(
                                *this, auxiliary, arg1_reg,
                                ptr[addressers.at(src2.traits->reg.getIdx())
                                        ->get_address(src2.offset * 4)]);
                            op_pair->plus(*this, dest, dest, auxiliary);
                        }
                        break;

                    case VECTOR_STRIDED:
                        if (src2.mask != vector_size)
                        {
                            ensure_initalized_mask(src2.mask);
                        }

                        kmovw(temp_k_mask, (src2.mask == vector_size)
                                               ? full_k_mask
                                               : tail_k_mask);

                        vgatherdps(arg2_register | temp_k_mask,
                                   ptr[addressers.at(src2.traits->reg.getIdx())
                                           ->get_address_without_index(
                                               src2.offset * 4) +
                                       tensor_strides[src2.traits->name]]);
                        if (op_pair->can_fuse())
                        {
                            op_pair->fuse(*this, C_VMMs[op.dest]++, arg1_reg,
                                          arg2_register);
                        }
                        else
                        {
                            Vmm auxiliary = Vmm(0);
                            Vmm dest      = C_VMMs[op.dest]++;

                            op_pair->multiplies(*this, auxiliary, arg1_reg,
                                                arg2_register);
                            op_pair->plus(*this, dest, dest, auxiliary);
                        }
                        break;
                    }
                }
            };
        }

        for (int off = 0; off < cycle; ++off, ++current)
        {
            issue_delayed_ops[current % cycle]();
        }

        for (auto const& p : addressers)
        {
            p.second->restore();
        }
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx2>> issue_unrolled_fmas(
        std::vector<fma_operation>                            fmas,
        std::map<int, std::shared_ptr<address_packer>> const& addressers,
        std::optional<int>                                    tail_mask)

    {
        if ((A_traits.access == SCALAR && B_traits.access == VECTOR_PACKED) ||
            (A_traits.access == VECTOR_PACKED && B_traits.access == SCALAR) ||
            (A_traits.access == VECTOR_PACKED &&
             B_traits.access == VECTOR_PACKED && C_traits.access != SCALAR))
        {
            issue_unrolled_fmas_scalar_vector();
            return;
        }

        // TODO(zi) Implement the new strategies when possible. With
        // all in-reg arguments for the FMAs

        // INPROGRESS - some analysis code is commented out below

        // std::map<memory_argument, int> freq;
        // for (auto const& f : fmas)
        // {
        //     ++freq[f.src1];
        //     ++freq[f.src2];
        // }

        // std::set<int> degrees;
        // for (auto const& p : freq)
        // {
        //     degrees.insert(p.second);
        // }

        // std::cout << "DEGREES:";
        // for (auto const d : degrees)
        //     std::cout << ' ' << d;
        // std::cout << std::endl;

        // Assignment of vector registers
        Vmm arg1_register, arg2_register;
        Vmm arg_A_strides, arg_B_strides;

        int next_vector_register = 0;

        arg1_register = Vmm(next_vector_register++);
        arg2_register = Vmm(next_vector_register++);

        std::map<std::string, Vmm> tensor_strides;

        if (A_traits.access == VECTOR_STRIDED)
        {
            tensor_strides["A"] = arg_A_strides = Vmm(next_vector_register++);
            vmovups(arg_A_strides, ptr[rip + A_access_strides_label]);
        }
        if (B_traits.access == VECTOR_STRIDED)
        {
            tensor_strides["B"] = arg_B_strides = Vmm(next_vector_register++);
            vmovups(arg_B_strides, ptr[rip + B_access_strides_label]);
        }

        Vmm ymm_full_mask, ymm_tail_mask, ymm_temp_register;

        bool requires_temp_ymm = false;

        if (A_traits.access == VECTOR_STRIDED ||
            B_traits.access == VECTOR_STRIDED)
        {
            ymm_full_mask = Vmm(next_vector_register++);
            vmovups(ymm_full_mask, ptr[rip + maskLabel_]);
            requires_temp_ymm = true;
        }

        // Needs a ymm register for the tail mask
        if (tail_mask &&
            (A_traits.access == VECTOR_STRIDED ||
             B_traits.access == VECTOR_STRIDED || C_traits.access == SCALAR))
        {
            // tail mask in ymm
            ymm_tail_mask     = Vmm(next_vector_register++);
            requires_temp_ymm = true;
        }

        // Need auxiliary ymm if not fused
        if (!op_pair->can_fuse())
        {
            requires_temp_ymm = true;
        }

        if (requires_temp_ymm)
        {
            ymm_temp_register = Vmm(next_vector_register++);
        }

        strong_assert(next_vector_register <= auxiliary_registers);

        most_frequent_queue<memory_argument> queue;

        for (auto const& p : addressers)
        {
            p.second->loop_prologue();
        }

        for (auto const& inst : fmas)
        {
            strong_assert(is_inside_current_limits(inst.coordinates));
            queue.inc(inst.src1);
            queue.inc(inst.src2);
        }

        int mask_size = -1;

        // TODO(zi) issue mask only when required (for avx512 as well)
        auto const ensure_initalized_mask = [&](int mask) {
            strong_assert(tail_mask && "Tail mask was not detected");
            strong_assert(mask == *tail_mask);

            if (mask_size == -1)
            {
                mask_size = mask;
                vmovups(ymm_tail_mask,
                        ptr[rip + maskLabel_ + 4 * (8 - mask_size)]);
            }
        };

        std::vector<Vmm> arg1_registers;
        arg1_registers.push_back(arg1_register);
        for (int i = first_unused_vmm_register;
             i < isa_traits<ISA>::total_vector_registers; ++i)
        {
            arg1_registers.push_back(Vmm(i));
        }

        // TODO(zi) replace this eyeballed value
        if (arg1_registers.size() > 5)
        {
            arg1_registers.resize(5);
        }

        int cycle = optim_config.delay_innermost_operations()
                        ? arg1_registers.size()
                        : 1;
        int current = 0;

        std::vector<std::function<void()>> issue_delayed_ops(cycle, []() {});

        for (; queue.size() > 0; ++current)
        {
            issue_delayed_ops[current % cycle]();

            auto arg1_reg = arg1_registers[current % cycle];

            auto addr = queue.top();
            queue.pop();

            LN_LOG(INFO) << tabs.back() << "LOAD " << addr.readable() << " ["
                         << addr.mask << "]\n";

            switch (addr.traits->access)
            {
            case SCALAR:
                if (C_traits.access == SCALAR && addr.mask != vector_size)
                {
                    ensure_initalized_mask(addr.mask);
                    vbroadcastss(arg1_reg,
                                 ptr[addressers.at(addr.traits->reg.getIdx())
                                         ->get_address(addr.offset * 4)]);

                    // reusing arg2_register as temporary
                    op_pair->set_to_identity(*this, arg2_register);
                    vblendvps(arg1_reg, arg1_reg, arg2_register, ymm_tail_mask);
                }
                else
                {
                    vbroadcastss(arg1_reg,
                                 ptr[addressers.at(addr.traits->reg.getIdx())
                                         ->get_address(addr.offset * 4)]);
                }
                break;

            case VECTOR_PACKED:
                if (C_traits.access == SCALAR && addr.mask != vector_size)
                {
                    op_pair->set_to_identity(*this, arg2_register);
                    ensure_initalized_mask(addr.mask);
                    vmaskmovps(arg1_reg, ymm_tail_mask,
                               ptr[addressers.at(addr.traits->reg.getIdx())
                                       ->get_address(addr.offset * 4)]);
                }
                else
                {
                    vmovups(arg1_reg,
                            ptr[addressers.at(addr.traits->reg.getIdx())
                                    ->get_address(addr.offset * 4)]);
                }
                break;

            case VECTOR_STRIDED:
                if (addr.mask != vector_size)
                {
                    ensure_initalized_mask(addr.mask);
                }
                if (addr.mask == vector_size)
                {
                    vmovups(ymm_temp_register, ymm_full_mask);
                }
                else
                {
                    vmovups(ymm_temp_register, ymm_tail_mask);
                    if (C_traits.access == SCALAR)
                    {
                        op_pair->set_to_identity(*this, arg1_reg);
                    }
                }

                vgatherdps(
                    arg1_reg,
                    ptr[addressers.at(addr.traits->reg.getIdx())
                            ->get_address_without_index(addr.offset * 4) +
                        tensor_strides[addr.traits->name]],
                    ymm_temp_register);
                break;
            }

            std::vector<fma_operation> delayed_fma_operations;

            for (auto it = fmas.begin(); it != fmas.end();)
            {
                if (is_inside_current_limits(it->coordinates) &&
                    (it->src1 == addr || it->src2 == addr))
                {
                    auto src1 = it->src1;
                    auto src2 = it->src2;
                    if (addr == it->src2)
                    {
                        std::swap(src1, src2);
                    }

                    queue.dec(src2);

                    delayed_fma_operations.push_back(*it);

                    LN_LOG(INFO) << tabs.back() << it->dest.readable()
                                 << " += " << it->src1.readable() << " * "
                                 << it->src2.readable() << "\n";
                    it = fmas.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            issue_delayed_ops[current % cycle] = [arg1_reg, addr,
                                                  delayed_fma_operations,
                                                  &addressers,
                                                  ymm_temp_register,
                                                  ymm_tail_mask, ymm_full_mask,
                                                  this, &ensure_initalized_mask,
                                                  arg2_register,
                                                  &tensor_strides]() {
                bool first = true;
                for (auto const& op : delayed_fma_operations)
                {
                    auto src1 = op.src1;
                    auto src2 = op.src2;

                    if (addr == op.src2)
                    {
                        std::swap(src1, src2);
                    }

                    if (first)
                    {
                        first = false;
                        addressers.at(src2.traits->reg.getIdx())
                            ->move_to(src2.offset * 4);
                    }

                    switch (src2.traits->access)
                    {
                    case SCALAR:
                        vbroadcastss(
                            arg2_register,
                            ptr[addressers.at(src2.traits->reg.getIdx())
                                    ->get_address(src2.offset * 4)]);

                        if (op_pair->can_fuse())
                        {
                            op_pair->fuse(*this, C_VMMs[op.dest]++, arg1_reg,
                                          arg2_register);
                        }
                        else
                        {
                            Vmm auxiliary = ymm_temp_register;
                            Vmm dest      = C_VMMs[op.dest]++;

                            op_pair->multiplies(*this, auxiliary, arg1_reg,
                                                arg2_register);
                            op_pair->plus(*this, dest, dest, auxiliary);
                        }
                        break;

                    case VECTOR_PACKED:
                        if (op_pair->can_fuse())
                        {
                            op_pair->fuse(
                                *this, C_VMMs[op.dest]++, arg1_reg,
                                ptr[addressers.at(src2.traits->reg.getIdx())
                                        ->get_address(src2.offset * 4)]);
                        }
                        else
                        {
                            Vmm auxiliary = ymm_temp_register;
                            Vmm dest      = C_VMMs[op.dest]++;

                            op_pair->multiplies(
                                *this, auxiliary, arg1_reg,
                                ptr[addressers.at(src2.traits->reg.getIdx())
                                        ->get_address(src2.offset * 4)]);
                            op_pair->plus(*this, dest, dest, auxiliary);
                        }
                        break;

                    case VECTOR_STRIDED:
                        if (src2.mask != vector_size)
                        {
                            ensure_initalized_mask(src2.mask);
                        }

                        vmovups(ymm_temp_register, (src2.mask == vector_size)
                                                       ? ymm_full_mask
                                                       : ymm_tail_mask);

                        vgatherdps(arg2_register,
                                   ptr[addressers.at(src2.traits->reg.getIdx())
                                           ->get_address_without_index(
                                               src2.offset * 4) +
                                       tensor_strides[src2.traits->name]],
                                   ymm_temp_register);
                        if (op_pair->can_fuse())
                        {
                            op_pair->fuse(*this, C_VMMs[op.dest]++, arg1_reg,
                                          arg2_register);
                        }
                        else
                        {
                            Vmm auxiliary = ymm_temp_register;
                            Vmm dest      = C_VMMs[op.dest]++;

                            op_pair->multiplies(*this, auxiliary, arg1_reg,
                                                arg2_register);
                            op_pair->plus(*this, dest, dest, auxiliary);
                        }
                        break;
                    }
                }
            };
        }

        for (int off = 0; off < cycle; ++off, ++current)
        {
            issue_delayed_ops[current % cycle]();
        }

        for (auto const& p : addressers)
        {
            p.second->restore();
        }
    }

    void issue_unrolled_fmas(
        std::vector<fma_operation>                            fmas,
        std::map<int, std::shared_ptr<address_packer>> const& addressers)
    {
        std::optional<int> tail_mask;

        auto update_tail_mask = [&](int m) {
            if (m != vector_size)
            {
                strong_assert(!tail_mask || *tail_mask == m);
                tail_mask = m;
            }
        };

        for (auto const& fma : fmas)
        {
            update_tail_mask(fma.src1.mask);
            update_tail_mask(fma.src2.mask);
        }

        issue_unrolled_fmas(fmas, addressers, tail_mask);
    }

    void issue_embedded_constants()
    {
        align_to(4);

        if (C_traits.access == VECTOR_STRIDED)
        {
            L(C_access_strides_label);
            for (int i = 0; i < vector_size; ++i)
            {
                dd(i * C_traits.innermost_stride * 4 /* bytes */);
            }
        }
        if (A_traits.access == VECTOR_STRIDED)
        {
            L(A_access_strides_label);
            for (int i = 0; i < vector_size; ++i)
            {
                dd(i * A_traits.innermost_stride * 4 /* bytes */);
            }
        }
        if (B_traits.access == VECTOR_STRIDED)
        {
            L(B_access_strides_label);
            for (int i = 0; i < vector_size; ++i)
            {
                dd(i * B_traits.innermost_stride * 4 /* bytes */);
            }
        }

        // elementwise tensors
        for (int i = 0; i < elementwise_traits.size(); i++)
        {
            tensor_traits elem_traits = elementwise_traits[i];
            if (elem_traits.access != VECTOR_STRIDED)
            {
                continue;
            }

            L(*(elementwise_labels[i]));
            for (int j = 0; j < vector_size; j++)
            {
                dd(j * elem_traits.innermost_stride * 4 /* bytes */);
            }
        }

        // Will be used as a mask for AVX2
        if (std::is_same_v<ISA, avx2>)
        {
            L(maskLabel_);
            for (int i = 0; i < 8; ++i)
            {
                dd(0xffffffff);
            }
            for (int i = 0; i < 8; ++i)
            {
                dd(0);
            }
        }
    }

    void issue_arithmetic_epilogue()
    {
        align_to(4);
        op_pair->issue_epilogue(*this);
    }

    void set_tensor_traits()
    {

        bool is_C_gathered =
            is_C_vectorized && C_strides.at(order.back().first) != 1;
        bool is_A_gathered =
            is_A_vectorized && A_strides.at(order.back().first) != 1;
        bool is_B_gathered =
            is_B_vectorized && B_strides.at(order.back().first) != 1;

        // Strides along the LSD dimension of the compute order
        int C_access_stride = 1;
        int A_access_stride = 1;
        int B_access_stride = 1;

        if (is_C_gathered)
        {
            C_access_stride = C_strides.at(order.back().first);
        }

        if (is_A_gathered)
        {
            A_access_stride = A_strides.at(order.back().first);
        }

        if (is_B_gathered)
        {
            B_access_stride = B_strides.at(order.back().first);
        }

        LN_LOG(DEBUG) << "C access stride: " << C_access_stride << "\n";
        LN_LOG(DEBUG) << "A access stride: " << A_access_stride << "\n";
        LN_LOG(DEBUG) << "B access stride: " << B_access_stride << "\n";

        access_kind C_access_kind =
            (is_C_vectorized ? (is_C_gathered ? VECTOR_STRIDED : VECTOR_PACKED)
                             : SCALAR);
        access_kind A_access_kind =
            (is_A_vectorized ? (is_A_gathered ? VECTOR_STRIDED : VECTOR_PACKED)
                             : SCALAR);
        access_kind B_access_kind =
            (is_B_vectorized ? (is_B_gathered ? VECTOR_STRIDED : VECTOR_PACKED)
                             : SCALAR);

        // TODO remove redundant information.
        C_traits = {"C",
                    C_access_kind,
                    CReg_,
                    &C_access_strides_label,
                    C_access_stride,
                    is_C_vectorized ? vector_size : 1};
        B_traits = {"B",
                    B_access_kind,
                    BReg_,
                    &B_access_strides_label,
                    B_access_stride,
                    is_B_vectorized ? vector_size : 1};
        A_traits = {"A",
                    A_access_kind,
                    AReg_,
                    &A_access_strides_label,
                    A_access_stride,
                    is_A_vectorized ? vector_size : 1};

        // Relaxed layout of C
        LN_LOG(DEBUG) << "C is "
                      << (C_traits.access == VECTOR_STRIDED ? "NOT " : "")
                      << "LSD packed\n";

        LN_LOG(DEBUG) << "C_access_len is: " << C_traits.access_len << "\n";
    }

    void set_elementwise_tensor_traits()
    {

        for (int i = 0; i < elementwise_strides.size(); i++)
        {
            std::map<std::string, int> strides = elementwise_strides[i];

            std::string name          = "elementwise_arg_" + std::to_string(i);
            bool        is_vectorized = is_C_vectorized;
            if ((strides.count(order.back().first) == 0) ||
                (strides.at(order.back().first) == 0))
            {
                // if the vectorized dimension doesn't exist
                // (either not specified or explicitly stated as 0)
                // for the followed elementwise tensors
                // we do scalar loads (potentially w/ broadcast)
                is_vectorized = false;
            }
            bool is_gathered =
                is_vectorized && strides.at(order.back().first) != 1;
            access_kind access =
                (is_vectorized ? (is_gathered ? VECTOR_STRIDED : VECTOR_PACKED)
                               : SCALAR);
            int access_stride =
                is_gathered ? strides.at(order.back().first) : 1;

            tensor_traits tt{name,
                             access,
                             elementwiseReg_[i],
                             elementwise_labels[i].get(),
                             access_stride,
                             is_vectorized ? vector_size : 1};
            elementwise_traits.push_back(tt);
        }
    }

    void set_available_vector_registers()
    {
        auxiliary_registers = 2;

        bool requires_temp_ymm = false;

        // Needs a ymm register for the tail mask used while loading
        // data from A and B.  Loads and stores to C can just reuse
        // the two reserved auxiliary registers.  This is needed when
        // either A or B are gathered - used in the gather
        // instruction, or when the compute requires a horizontal - to
        // make sure we don't accumulate to masked out lanes of the
        // vector register.

        if ((B_traits.access == VECTOR_STRIDED ||
             A_traits.access == VECTOR_STRIDED || C_traits.access == SCALAR) &&
            (sizes.at(vectorized_var) % vector_size) &&
            std::is_same_v<ISA, avx2>)
        {
            // tail mask in ymm
            ++auxiliary_registers;    // Stores tail mask
            requires_temp_ymm = true; // Temp for gathers or
            LN_LOG(DEBUG) << "Requires extra ymm for a tail mask (total: "
                          << auxiliary_registers << ")\n";
        }

        if (A_traits.access == VECTOR_STRIDED ||
            B_traits.access == VECTOR_STRIDED)
        {
            if (A_traits.access == VECTOR_STRIDED)
                ++auxiliary_registers; // A strides
            if (B_traits.access == VECTOR_STRIDED)
                ++auxiliary_registers; // B strides

            LN_LOG(DEBUG) << "Requires VMMs for A,B strides (total: "
                          << auxiliary_registers << ")\n";

            if (std::is_same_v<ISA, avx2>)
            {
                // Needs a full mask
                ++auxiliary_registers;    // Stores full mask (all 0xff)
                requires_temp_ymm = true; // Temp for gathers
            }
        }

        if (std::is_same_v<ISA, avx2> && !op_pair->can_fuse())
        {
            // keep around temporary register when not fused
            requires_temp_ymm = true;
        }

        if (requires_temp_ymm)
        {
            ++auxiliary_registers;
        }

        if ((std::is_same_v<ISA, avx512> ||
             std::is_same_v<ISA, avx2_plus>)&&!op_pair->can_fuse())
        {
            // we don't have a temporary ymm, so we
            // explicitly increase auxiliary registers
            ++auxiliary_registers;
        }

        // END LOGIC FOR INNERMOST LOOP

        {
            int auxiliary_registers_for_C_access = 0;
            if (C_traits.access == VECTOR_STRIDED)
            {
                ++auxiliary_registers_for_C_access;
            }

            if (std::is_same_v<ISA, avx2>)
            {
                if (C_traits.access == VECTOR_STRIDED)
                {
                    auxiliary_registers_for_C_access +=
                        2; // Full mask and temp register
                }

                if (C_traits.access_len == vector_size &&
                    sizes.at(vectorized_var) % vector_size)
                {
                    ++auxiliary_registers_for_C_access; // tail mask
                }
            }

            auxiliary_registers =
                std::max(auxiliary_registers, auxiliary_registers_for_C_access);

            // Horizontal stores require adds
            if (C_traits.access == SCALAR)
            {
                // This is noop, but here for logical clarity
                auxiliary_registers = std::max(auxiliary_registers, 2);
            }
        }

        available_registers =
            isa_traits<ISA>::total_vector_registers - auxiliary_registers;

        LN_LOG(DEBUG) << "AVAILABLE REGS: " << available_registers << "\n";
    }

    void set_in_register_tensor_pointers()
    {
        in_register_tensor_pointers.push_back({"A_Tensor", AReg_, A_strides});
        in_register_tensor_pointers.push_back({"B_Tensor", BReg_, B_strides});
        in_register_tensor_pointers.push_back({"C_Tensor", CReg_, C_strides});
    }

    void set_in_register_elementwise_tensor_pointers()
    {
        for (int i = 0; i < elementwise_strides.size(); i++)
        {
            std::map<std::string, int> strides = elementwise_strides[i];
            tensor_traits              traits  = elementwise_traits[i];
            in_register_tensor_pointers.push_back(
                {traits.name, traits.reg, strides});
        }
    }

    // Returns the first loop that can hold C in register file, and
    // the first loop to be unrolled.
    std::tuple<int, int, int> possibly_inject_a_loop()
    {

        auto padded_sizes = sizes;
        padded_sizes[vectorized_var] =
            round_up(padded_sizes[vectorized_var], vector_size);

        std::map<std::string, std::vector<int>> ranges;
        for (auto const& p : padded_sizes)
        {
            ranges[p.first].push_back(p.second);
        }

        int registers_required =
            std::accumulate(padded_sizes.begin(), padded_sizes.end(), 1,
                            [&](int v, auto const& s) {
                                return C_formula.count(s.first) ? v * s.second
                                                                : v;
                            }) /
            C_traits.access_len;

        std::int64_t total_required_fma_operations =
            std::accumulate(padded_sizes.begin(), padded_sizes.end(),
                            (std::int64_t)1,
                            [&](std::int64_t v, auto const& s) {
                                // std::cout << v << " :: " << s.second << "\n";
                                return (B_strides.count(s.first) ||
                                        A_strides.count(s.first) ||
                                        C_strides.count(s.first))
                                           ? v * s.second
                                           : v;
                            }) /
            vector_size;

        LN_LOG(DEBUG) << "REGS REQUIRED: " << registers_required
                      << " FMAS: " << total_required_fma_operations << "\n";

        int first_loop_that_can_hold_C = 0;

        LN_LOG(DEBUG) << "Registers originally required: " << registers_required
                      << "\n";
        LN_LOG(DEBUG) << "C_access_len: " << C_traits.access_len << "\n";

        // auto sizes_copy = padded_sizes;

        auto it_end = --(order.end());
        auto it     = order.begin();

        for (; registers_required > available_registers && it != it_end; ++it)
        {
            if (C_formula.count(it->first))
            {
                if (is_C_vectorized && it->first == vectorized_var)
                {
                    registers_required /=
                        (ranges[it->first].back() / vector_size);
                    registers_required *= (it->second / vector_size);
                }
                else
                {
                    registers_required /= ranges[it->first].back();
                    registers_required *= it->second;
                }
            }

            if (it->first == vectorized_var)
            {
                total_required_fma_operations /=
                    ceil_div(ranges[it->first].back(), vector_size);
                total_required_fma_operations *= (it->second / vector_size);
            }
            else
            {
                total_required_fma_operations /= ranges[it->first].back();
                total_required_fma_operations *= it->second;
            }

            ++first_loop_that_can_hold_C;

            LN_LOG(DEBUG) << "    AT LOOP " << first_loop_that_can_hold_C
                          << " REGS REQUIRED: " << registers_required
                          << " FMAS: " << total_required_fma_operations << "\n";

            ranges[it->first].push_back(it->second);
        }

        // This will happen only when the deepest loop in the nest
        // does iterate over C, otherwise, we'd have to converge
        // to register_required = 1 at some point.
        if (registers_required > available_registers)
        {
            strong_assert(it == it_end);
            while (C_formula.count(it->first) == 0 && it != order.begin())
            {
                ranges[it->first].pop_back();

                if (it->first == vectorized_var)
                {
                    total_required_fma_operations /= (it->second / vector_size);
                    total_required_fma_operations *=
                        ceil_div(ranges[it->first].back(), vector_size);
                }
                else
                {
                    total_required_fma_operations /= it->second;
                    total_required_fma_operations *= ranges[it->first].back();
                }

                --it;
                --first_loop_that_can_hold_C;
            }

            // strong_assert(it_end != order.begin());

            auto pair = *it;

            int register_limit =
                (it == it_end ? std::min(available_registers, max_fmas_unrolled)
                              : available_registers);

            // TODO(zi) MAYBE - increase max_fmas_unrolled to
            // available_registers?  There's probably never need
            // to request smaller unroll amount than the number of
            // available registers.
            pair.second =
                register_limit *
                ((pair.first == vectorized_var && is_C_vectorized) ? vector_size
                                                                   : 1);

            registers_required = register_limit;

            LN_LOG(DEBUG) << "INJECTING A LOOP: " << pair.first << ", "
                          << pair.second << "\n";
            it = order.insert(it, pair);

            if (it->first == vectorized_var)
            {
                total_required_fma_operations /=
                    ceil_div(ranges[it->first].back(), vector_size);
                total_required_fma_operations *= (it->second / vector_size);
            }
            else
            {
                total_required_fma_operations /= ranges[it->first].back();
                total_required_fma_operations *= it->second;
            }

            ++first_loop_that_can_hold_C;

            LN_LOG(DEBUG) << "REVISED AT LOOP " << first_loop_that_can_hold_C
                          << " REGS REQUIRED: " << registers_required
                          << " FMAS: " << total_required_fma_operations << "\n";

            ranges[it->first].push_back(it->second);

            ++nest_depth;
        }

        int first_unrolled_loop = first_loop_that_can_hold_C;

        it_end = --(order.end());

        for (;
             total_required_fma_operations > max_fmas_unrolled && it != it_end;
             ++it)
        {
            if (it->first == vectorized_var)
            {
                total_required_fma_operations /=
                    ceil_div(ranges[it->first].back(), vector_size);
                total_required_fma_operations *= (it->second / vector_size);
            }
            else
            {
                total_required_fma_operations /= ranges[it->first].back();
                total_required_fma_operations *= it->second;
            }

            ++first_unrolled_loop;

            LN_LOG(DEBUG) << "   AT LOOP " << first_unrolled_loop
                          << " FMAS: " << total_required_fma_operations << "\n";

            ranges[it->first].push_back(it->second);
        }

        if (total_required_fma_operations > max_fmas_unrolled)
        {
            auto pair = *it;

            pair.second                   = max_fmas_unrolled * vector_size;
            total_required_fma_operations = max_fmas_unrolled;
            ++first_unrolled_loop;

            LN_LOG(DEBUG) << "INJECTING A LOOP (for unroll): " << pair.first
                          << ", " << pair.second << "\n";
            order.insert(it, pair);
            ++nest_depth;
        }

        return {first_loop_that_can_hold_C, first_unrolled_loop,
                total_required_fma_operations};
    }

    int assign_vmm_registers(int depth_for_register_blocked_C,
                             int inner_fma_operations)
    {
        auto collected_load_store =
            collect_default_loads_and_stores_at(depth_for_register_blocked_C);

        // Assign Vector registers to hold C block
        // TODO(zi) better heuristics here
        {
            int next         = auxiliary_registers;
            int per_register = 1;

            if (collected_load_store.size() < available_registers &&
                inner_fma_operations > 48 &&
                optim_config.split_vector_registers())
            {
                per_register =
                    available_registers / collected_load_store.size();
                per_register = std::min(per_register, 3);
            }

            for (auto const& c : collected_load_store)
            {
                LN_LOG(DEBUG) << "LOAD/STORE: " << c.readable() << " ("
                              << per_register << " VMMs)\n";
                C_VMMs[c] = multi_zmm(per_register, next);
                next += per_register;
            }

            strong_assert(next <= isa_traits<ISA>::total_vector_registers);

            return next;
        }
    }

    void initialize_loops_data()
    {

        // Initialize the outermost limits
        for (auto const& p : sizes)
        {
            limits[p.first].push_back(p.second);
        }

        // Compute the ranges of all loops
        {
            auto sizes_copy = sizes;
            for (auto const& o : order)
            {
                loops.push_back({o.first, sizes_copy[o.first], o.second});
                sizes_copy[o.first] = o.second;

                print_ld(loops.back());
            }
        }

        bool first = true;
        for (int i = 0; i < loops.size() - 1; ++i)
        {
            if (loops[i].var == vectorized_var)
            {
                if (first)
                {
                    first = false;
                }
                else
                {
                    // std::cout << loops[i].var << " :: " << loops[i].end
                    //          << std::endl;
                    strong_assert((loops[i].end % vector_size) == 0);
                }
            }
        }
    }

    std::tuple<int, int> lower_register_blocked_loop(int first_unrolled_loop,
                                                     int inner_fma_operations)
    {
        int first_loop_that_can_hold_C = first_unrolled_loop;
        while (first_loop_that_can_hold_C > 0 &&
               C_formula.count(loops[first_loop_that_can_hold_C - 1].var) == 0)
        {
            --first_loop_that_can_hold_C;
            // TODO(zi) check math
            auto const& loop      = loops[first_loop_that_can_hold_C];
            int         expansion = loop.end / loop.delta;
            inner_fma_operations *= expansion;
        }

        LN_LOG(DEBUG) << "LOAD/STORE C MOVED TO LOOP: "
                      << first_loop_that_can_hold_C << " OVER "
                      << loops[first_loop_that_can_hold_C].var << " WITH "
                      << inner_fma_operations << " INNER FMAs\n";

        return {first_loop_that_can_hold_C, inner_fma_operations};
    }

    std::map<int, std::shared_ptr<address_packer>> create_null_addressers()
    {
        strong_assert(!optim_config.use_address_packer());

        std::map<int, std::shared_ptr<address_packer>> addressers;

        addressers[AReg_.getIdx()] =
            std::make_shared<null_address_packer>(this, AReg_);
        addressers[BReg_.getIdx()] =
            std::make_shared<null_address_packer>(this, BReg_);

        return addressers;
    }

    std::map<int, std::shared_ptr<address_packer>>
    create_addressers(std::vector<fma_operation> unrolled_fmas)
    {
        if (!optim_config.use_address_packer())
        {
            return create_null_addressers();
        }

        struct address_load
        {
            int   offset;
            int   len;
            Reg64 reg;

            bool operator<(address_load const& b) const
            {
                int my_reg    = reg.getIdx();
                int other_reg = b.reg.getIdx();
                return std::tie(offset, len, my_reg) <
                       std::tie(b.offset, b.len, other_reg);
            }
        };

        std::map<address_load, int>                    global_patterns;
        std::map<int, std::shared_ptr<address_packer>> addressers;
        most_frequent_queue<memory_argument>           queue;

        auto unrolled_fmas_copy = unrolled_fmas;

        for (auto const& inst : unrolled_fmas)
        {
            queue.inc(inst.src1);
            queue.inc(inst.src2);
        }

        while (queue.size() > 0)
        {
            auto addr = queue.get_top_then_pop();

            std::vector<address_load> addresses;

            for (auto it = unrolled_fmas_copy.begin();
                 it != unrolled_fmas_copy.end();)
            {
                if (it->src1 == addr || it->src2 == addr)
                {
                    auto src1 = it->src1;
                    auto src2 = it->src2;
                    if (addr == it->src2)
                    {
                        std::swap(src1, src2);
                    }

                    queue.dec(src2);

                    addresses.push_back({src2.offset * 4,
                                         src2.traits->access_len,
                                         src2.traits->reg});

                    it = unrolled_fmas_copy.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            // Find patterns
            std::sort(addresses.begin(), addresses.end());

            std::map<address_load, int> patterns;

            for (std::size_t i = 1; i < addresses.size(); ++i)
            {
                if (addresses[i].len == addresses[0].len)
                {
                    auto diff = addresses[i].offset - addresses[0].offset;

                    if (diff != 0)
                    {
                        auto it = std::find_if(patterns.begin(), patterns.end(),
                                               [&](auto const& p) {
                                                   return (diff %
                                                           p.first.offset) == 0;
                                               });

                        if (it != patterns.end())
                        {
                            ++it->second;
                        }
                        else
                        {
                            patterns[address_load{diff, addresses[0].len,
                                                  addresses[0].reg}] = 1;
                        }
                    }
                }
            }

            for (auto const& p : patterns)
            {
                global_patterns[p.first] =
                    std::max(p.second, global_patterns[p.first]);
                LN_LOG(INFO) << "> PATTERN: " << p.first.offset << " ["
                             << p.first.len << "] "
                             << " of " << p.second << "\n";
            }
        }

        addressers[AReg_.getIdx()] =
            std::make_shared<null_address_packer>(this, AReg_);
        addressers[BReg_.getIdx()] =
            std::make_shared<null_address_packer>(this, BReg_);

        if (global_patterns.size() == 1)
        {
            auto& p = *global_patterns.begin();
            if (p.second > 18) // TODO(zi) better heuristic
            {
                addressers[p.first.reg.getIdx()] =
                    std::make_shared<double_base_SIB_address_packer>(
                        this, p.first.reg, p.second, p.first.offset,
                        addressing_registers);
            }
            else if (p.second > 4)
            {
                addressers[p.first.reg.getIdx()] =
                    std::make_shared<simple_SIB_address_packer>(
                        this, p.first.reg, p.second, p.first.offset,
                        addressing_registers);
            }
        }

        return addressers;
    }

    void issue_loop_helper(
        int depth, bool save_loop, bool save_ptrs,
        int depth_for_register_blocked_C, int unroll_stage,
        std::map<int, std::shared_ptr<address_packer>> const& addressers,
        bool issue_first_alpha_logic, int max_alpha, bool issue_max_alpha_logic)
    {
        LN_LOG(INFO) << tabs.back() << "// DEPTH: " << depth
                     << " MAX_ALPHA: " << max_alpha << "\n";

        std::vector<fma_operation> unrolled_fmas;
        std::set<memory_argument>  collected_load_store;

        if (depth == depth_for_register_blocked_C)
        {
            collected_load_store = collect_loads_and_stores_below(depth);
            issue_C_loads(collected_load_store, issue_first_alpha_logic);
        }

        if (depth == unroll_stage)
        {
            unrolled_fmas = collect_unrolled_FMAs_below(depth);
            issue_unrolled_fmas(unrolled_fmas, addressers);
        }
        else
        {
            auto const& loop = loops[depth];

            std::string var_name = loop.var + "_" + std::to_string(loop.delta);

            auto loop_end        = limits[loop.var].back();
            auto full_iterations = loop_end / loop.delta;
            auto tail            = loop_end % loop.delta;

            bool multiple_iterations =
                (full_iterations > 1) || ((full_iterations == 1) && tail);

            if (multiple_iterations && save_ptrs)
            {
                push_pointers(loop.var);
            }

            if (full_iterations > 1 && save_loop)
            {
                push(loopReg_);
            }

            if (full_iterations > 0)
            {
                LN_LOG(INFO)
                    << tabs.back() << "FOR: " << var_name << " FROM 0 TO "
                    << loop_end << " BY " << loop.delta << " {\n";
            }

            // TODO (zi): Optimize so that if tail exists, the post-op
            // logic is only generated in the tail.
            int new_max_alpha = max_alpha;

            bool recursive_issue_max_alpha_logic = issue_max_alpha_logic;
            if (tail && depth < depth_for_register_blocked_C &&
                C_formula.count(loop.var) == 0)
            {
                recursive_issue_max_alpha_logic = false;
            }

            if (full_iterations > 1)
            {
                mov(loopReg_.cvt32(), full_iterations);
                Label loopLabel;
                L(loopLabel);

                // --------------------------------------------------
                // RECURSION
                if (depth < depth_for_register_blocked_C &&
                    C_formula.count(loop.var) == 0)
                {
                    new_max_alpha += (full_iterations - 1 + (tail ? 1 : 0)) * 2;
                }

                limits[loop.var].push_back(loop.delta);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_helper(
                    depth + 1, true, true, depth_for_register_blocked_C,
                    unroll_stage, addressers, issue_first_alpha_logic,
                    new_max_alpha, recursive_issue_max_alpha_logic);
                tabs.pop_back();
                limits[loop.var].pop_back();
                // --------------------------------------------------
                // RECURSION

                advance_pointers(loop.var, loop.delta);

                if (depth < depth_for_register_blocked_C &&
                    C_formula.count(loop.var) == 0)
                {
                    add(AlphaReg_, 2);
                }

                dec(loopReg_.cvt32());
                jnz(loopLabel);
            }
            else if (full_iterations == 1)
            {
                // --------------------------------------------------
                // RECURSION
                if (tail && depth < depth_for_register_blocked_C &&
                    C_formula.count(loop.var) == 0)
                {
                    new_max_alpha += 2;
                }

                limits[loop.var].push_back(loop.delta);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_helper(depth + 1, save_loop, (tail > 0) || save_ptrs,
                                  depth_for_register_blocked_C, unroll_stage,
                                  addressers, issue_first_alpha_logic,
                                  new_max_alpha,
                                  recursive_issue_max_alpha_logic);
                tabs.pop_back();
                limits[loop.var].pop_back();
                // --------------------------------------------------
                // RECURSION

                if (tail)
                {
                    if (depth < depth_for_register_blocked_C &&
                        C_formula.count(loop.var) == 0)
                    {
                        add(AlphaReg_, 2);
                    }

                    advance_pointers(loop.var, loop.delta);
                }
            }

            if (tail)
            {
                bool recursive_issue_first_alpha_logic =
                    issue_first_alpha_logic;
                if (depth < depth_for_register_blocked_C &&
                    C_formula.count(loop.var) == 0 && full_iterations >= 1)
                {
                    recursive_issue_max_alpha_logic = false;
                }

                LN_LOG(INFO) << tabs.back() << "TAIL: " << var_name << " OF "
                             << tail << " {\n";

                limits[loop.var].push_back(tail);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_helper(
                    depth + 1, save_loop, save_ptrs,
                    depth_for_register_blocked_C, unroll_stage, addressers,
                    recursive_issue_first_alpha_logic, new_max_alpha,
                    issue_max_alpha_logic); // TODO(zi) something is weird with
                                            // this logic, should work with
                                            // !multiple_iterations && save_ptrs
                tabs.pop_back();
                limits[loop.var].pop_back();

                LN_LOG(INFO) << tabs.back() << "} END TAIL\n";
            }

            if (full_iterations > 0)
            {
                LN_LOG(INFO) << tabs.back() << "} END FOR\n";
            }

            if (multiple_iterations && depth < depth_for_register_blocked_C &&
                C_formula.count(loop.var) == 0)
            {
                LN_LOG(INFO)
                    << tabs.back() << "SUB LOCKER: " << full_iterations << "\n";
                sub(AlphaReg_, full_iterations * 2);
            }

            if (full_iterations > 1 && save_loop)
            {
                pop(loopReg_);
            }

            if (multiple_iterations && save_ptrs)
            {
                pop_pointers(loop.var);
            }
        }

        if (depth == depth_for_register_blocked_C)
        {
            issue_C_stores(collected_load_store, max_alpha,
                           issue_max_alpha_logic);
        }
    }

    void issue_loops(
        int depth_for_register_blocked_C, int unroll_stage,
        std::map<int, std::shared_ptr<address_packer>> const& addressers)
    {
        issue_loop_helper(0, false, false, depth_for_register_blocked_C,
                          unroll_stage, addressers, true, 1, true);
    }

    void issue_unrolled_fmas_dry_run(std::vector<fma_operation> fmas,
                                     int                        num_iterations)
    {
        // List of usages

        int src1_reg = fmas[0].src1.traits->reg.getIdx();
        int src2_reg = fmas[0].src2.traits->reg.getIdx();

        std::map<tensor_location_t, std::deque<int>> remaining_usages;

        for (int i = 0; i < fmas.size(); ++i)
        {
            remaining_usages[{src1_reg, fmas[i].src1.offset * 4}].push_back(i);
            remaining_usages[{src2_reg, fmas[i].src2.offset * 4}].push_back(i);
        }

        auto num_regs =
            isa_traits<ISA>::total_vector_registers - first_unused_vmm_register;

        std::deque<int> free_regs;

        for (int i = op_pair->can_fuse() ? 0 : 1; i < auxiliary_registers; ++i)
        {
            free_regs.push_back(i);
        }

        for (auto i = 0; i < num_regs; ++i)
        {
            free_regs.push_back(first_unused_vmm_register + i);
        }

        struct table_entry
        {
            tensor_location_t tensor_location;

            int vmm_idx;
            int next_usage;
        };

        using namespace boost::multi_index;

        // clang-format off

        using table_container = multi_index_container<
            table_entry,
            indexed_by<
                ordered_unique<
                    member<table_entry,
                           tensor_location_t,
                           &table_entry::tensor_location>
                    >,
                ordered_non_unique<
                    member<table_entry, int, &table_entry::vmm_idx>
                    >,
                ordered_non_unique<
                    member<table_entry, int, &table_entry::next_usage>,
                    std::greater<int>
                    >
                >
            >;

        // clang-format on

        table_container table;

        auto& tensor_location_index = table.template get<0>();
        auto& vreg_index            = table.template get<1>();
        auto& next_usage_index      = table.template get<2>();

        std::vector<instruction_t> instructions;

        auto add_load_instruction = [&](int vmm_idx, int tensor_idx, int offset,
                                        int does_broadcast) {
            load_instruction insn{
                vmm_idx, does_broadcast, {tensor_idx, offset}};

            strong_assert(remaining_usages.count(insn.tensor_loc) &&
                          remaining_usages[insn.tensor_loc].size() > 0);

            int next_usage = remaining_usages[insn.tensor_loc].front();

            table.insert(table_entry{insn.tensor_loc, vmm_idx, next_usage});

            instructions.push_back(insn);
        };

        auto free_a_register = [&]() {
            auto nu_it = next_usage_index.begin();
            strong_assert(nu_it != next_usage_index.end());

            int reg_no = nu_it->vmm_idx;

            auto it = vreg_index.find(reg_no);
            while (it != vreg_index.end())
            {
                vreg_index.erase(it);
                it = vreg_index.find(reg_no);
            }

            return reg_no;
        };

        auto maybe_issue_load = [&](tensor_location_t const& mem_location,
                                    bool does_broadcast, bool folding_allowed) {
            if (auto it = tensor_location_index.find(mem_location);
                it == tensor_location_index.end())
            {
                if (folding_allowed &&
                    remaining_usages[mem_location].size() == 1)
                {
                    return;
                }

                int r = 0;
                if (free_regs.size())
                {
                    r = free_regs.front();
                    free_regs.pop_front();
                }
                else
                {
                    r = free_a_register();
                }

                add_load_instruction(r, mem_location.idx, mem_location.offset,
                                     does_broadcast);
            }
        };

        auto mark_usage = [&](tensor_location_t const& mem_location) {
            remaining_usages[mem_location].pop_front();
            if (auto it = tensor_location_index.find(mem_location);
                it != tensor_location_index.end())
            {
                auto v = *it;
                tensor_location_index.erase(it);
                if (remaining_usages[mem_location].size())
                {
                    v.next_usage = remaining_usages[mem_location].front();
                    table.insert(v);
                }
                else
                {
                    remaining_usages.erase(mem_location);
                    free_regs.push_back(v.vmm_idx);
                }
            }
        };

        for (int i = 0; i < fmas.size(); ++i)
        {
            tensor_location_t left_loc  = {src1_reg, fmas[i].src1.offset * 4};
            tensor_location_t right_loc = {src2_reg, fmas[i].src2.offset * 4};

            if (auto it = tensor_location_index.find(left_loc);
                it == tensor_location_index.end() &&
                remaining_usages[left_loc].size() == 1)
            {
                if (fmas[i].src1.traits->access != SCALAR ||
                    !std::is_same_v<ISA, avx2>)
                {
                    std::swap(left_loc, right_loc);
                    std::swap(fmas[i].src1, fmas[i].src2);
                }
            }

            maybe_issue_load(left_loc, fmas[i].src1.traits->access == SCALAR,
                             false);
            maybe_issue_load(right_loc, fmas[i].src2.traits->access == SCALAR,
                             fmas[i].src2.traits->access != SCALAR ||
                                 !std::is_same_v<ISA, avx2>);

            auto left_it  = tensor_location_index.find(left_loc);
            auto right_it = tensor_location_index.find(right_loc);

            strong_assert(left_it != tensor_location_index.end());
            strong_assert((right_it != tensor_location_index.end()) ||
                          remaining_usages[right_loc].size() == 1);

            if (right_it != tensor_location_index.end())
            {
                instructions.push_back(
                    fmla_instruction{(int)((C_VMMs[fmas[i].dest]++).getIdx()),
                                     left_it->vmm_idx, right_it->vmm_idx});
            }
            else
            {
                instructions.push_back(fmla_instruction{
                    (int)((C_VMMs[fmas[i].dest]++).getIdx()), left_it->vmm_idx,
                    memory_src{right_loc.idx, right_loc.offset,
                               fmas[i].src2.traits->access == SCALAR}});
            }

            mark_usage(left_loc);
            mark_usage(right_loc);
        }

        // for (int i = 0; i < fmas.size(); ++i)
        // {
        //     tensor_location_t scalar_loc = {src1_reg, fmas[i].src1.offset *
        //     4}; tensor_location_t vector_loc = {src2_reg, fmas[i].src2.offset
        //     * 4};

        //     if (auto it = tensor_location_index.find(scalar_loc);
        //         it == tensor_location_index.end())
        //     {
        //         if (free_regs.size() == 0)
        //         {
        //             free_regs.push_back(free_a_register());
        //         }

        //         load_scalar(free_regs.front(), src1_reg,
        //                     fmas[i].src1.offset * 4);
        //         free_regs.pop_front();

        //         strong_assert(tensor_location_index.find(scalar_loc) !=
        //                tensor_location_index.end());
        //     }

        //     if (auto it = tensor_location_index.find(vector_loc);
        //         it == tensor_location_index.end())
        //     {
        //         if (free_regs.size() == 0)
        //         {
        //             free_regs.push_back(free_a_register());
        //         }

        //         load_vector(free_regs.front(), src2_reg,
        //                     fmas[i].src2.offset * 4);
        //         free_regs.pop_front();

        //         strong_assert(tensor_location_index.find(vector_loc) !=
        //                tensor_location_index.end());
        //     }

        //     auto s_it = tensor_location_index.find(scalar_loc);
        //     auto v_it = tensor_location_index.find(vector_loc);

        //     strong_assert(s_it != tensor_location_index.end());
        //     strong_assert(v_it != tensor_location_index.end());

        //     // issue FMA
        //     instructions.push_back(fmla_instruction{
        //         {(int)((C_VMMs[fmas[i].dest]++).getIdx()), vector_size},
        //         {v_it->vreg_idx, v_it->vreg_lane},
        //         {s_it->vreg_idx, s_it->vreg_lane}}); // update datastructures

        //     strong_assert(remaining_usages.count(scalar_loc) &&
        //            remaining_usages[scalar_loc].size() &&
        //            remaining_usages[scalar_loc].front() == s_it->next_usage);

        //     strong_assert(remaining_usages.count(vector_loc) &&
        //            remaining_usages[vector_loc].size() &&
        //            remaining_usages[vector_loc].front() == v_it->next_usage);

        //     // Update scalar
        //     {
        //         auto s = *s_it;
        //         tensor_location_index.erase(s_it);
        //         remaining_usages[scalar_loc].pop_front();
        //         if (remaining_usages[scalar_loc].size())
        //         {
        //             s.next_usage = remaining_usages[scalar_loc].front();
        //             table.insert(s);
        //         }
        //         else
        //         {
        //             remaining_usages.erase(scalar_loc);
        //             if (vreg_index.find(s.vreg_idx) == vreg_index.end())
        //             {
        //                 free_regs.push_back(s.vreg_idx);
        //             }
        //         }
        //     }

        //     // Update vector
        //     {
        //         auto v = *v_it;
        //         tensor_location_index.erase(v_it);
        //         remaining_usages[vector_loc].pop_front();
        //         if (remaining_usages[vector_loc].size())
        //         {
        //             v.next_usage = remaining_usages[vector_loc].front();
        //             table.insert(v);
        //         }
        //         else
        //         {
        //             remaining_usages.erase(vector_loc);
        //             if (vreg_index.find(v.vreg_idx) == vreg_index.end())
        //             {
        //                 free_regs.push_back(v.vreg_idx);
        //             }
        //         }
        //     }
        // }

        // // {
        // //     std::map<int, int> tensor_offsets;

        // //     for (auto const& insn : instructions)
        // //     {
        // //         std::visit(
        // //             overloaded{
        // //                 [&](load_instruction const& i) {
        // //                     int ptr_reg_idx =
        // i.tensor_location.tensor_idx;
        // //                     int ptr_offset  =
        // //                     i.tensor_location.tensor_offset;

        // //                     auto delta =
        // //                         ptr_offset - tensor_offsets[ptr_reg_idx];

        // //                     tensor_offsets[ptr_reg_idx] = ptr_offset;

        // //                     sadd_freq[delta] += num_iterations;
        // //                 },
        // //                 [&](fmla_instruction const&) {},
        // [](std::monostate)
        // //                 {}},
        // //             insn);
        // //     }
        // // }

        // // Move loads
        // for (int i = 1; i < instructions.size(); ++i)
        // {
        //     if (std::holds_alternative<load_instruction>(instructions[i]))
        //     {
        //         auto load = std::get<load_instruction>(instructions[i]);
        //         for (int pos = i, moves = 0; pos > 0 && moves < 10;
        //              --pos, ++moves)
        //         {
        //             if (std::holds_alternative<fmla_instruction>(
        //                     instructions[pos - 1]))
        //             {
        //                 auto fma =
        //                     std::get<fmla_instruction>(instructions[pos -
        //                     1]);
        //                 if (load.vreg != fma.left_src.number &&
        //                     load.vreg != fma.right_src.number)
        //                 {
        //                     std::swap(instructions[pos], instructions[pos -
        //                     1]);
        //                 }
        //                 else
        //                 {
        //                     pos = 1; // break;
        //                 }
        //             }
        //             else
        //             {
        //                 pos = 1; // break
        //             }
        //         }
        //     }
        // }

        // // Change offset to post increment
        // {
        //     std::map<int, int> reg_to_location;

        //     for (auto it = instructions.rbegin(); it != instructions.rend();
        //          ++it)
        //     {
        //         auto& insn = *it;
        //         if (std::holds_alternative<load_instruction>(insn))
        //         {
        //             auto& load = std::get<load_instruction>(insn);
        //             if
        //             (reg_to_location.count(load.tensor_location.tensor_idx))
        //             {
        //                 auto delta =
        //                     reg_to_location[load.tensor_location.tensor_idx]
        //                     - load.tensor_location.tensor_offset;

        //                 reg_to_location[load.tensor_location.tensor_idx] =
        //                     load.tensor_location.tensor_offset;
        //                 load.tensor_location.tensor_offset = delta;

        //                 sadd_freq[delta] += num_iterations;
        //             }
        //             else
        //             {
        //                 reg_to_location[load.tensor_location.tensor_idx] =
        //                     load.tensor_location.tensor_offset;
        //                 load.tensor_location.tensor_offset = 0;
        //             }
        //         }
        //     }
        // }

        constexpr int max_load_moves = 16;

        // Move loads
        for (int i = 1; i < instructions.size(); ++i)
        {
            if (std::holds_alternative<load_instruction>(instructions[i]))
            {
                auto load = std::get<load_instruction>(instructions[i]);

                if (load_offsets_first_pass)
                {
                    load_offsets[load.tensor_loc.idx].push_back(
                        load.tensor_loc.offset);
                }

                for (int pos = i, moves = 0; pos > 0 && moves < max_load_moves;
                     --pos, ++moves)
                {
                    if (std::holds_alternative<fmla_instruction>(
                            instructions[pos - 1]))
                    {
                        auto fma =
                            std::get<fmla_instruction>(instructions[pos - 1]);
                        if (load.vmm_idx != fma.left_src &&
                            (std::holds_alternative<memory_src>(
                                 fma.right_src) ||
                             (std::get<int>(fma.right_src) != load.vmm_idx)))
                        {
                            std::swap(instructions[pos], instructions[pos - 1]);
                        }
                        else
                        {
                            pos = 1; // break;
                        }
                    }
                    else
                    {
                        pos = 1; // break
                    }
                }
            }
        }

        load_offsets_first_pass = false;

        instruction_IRs.push_back(std::move(instructions));
    }

    void issue_loop_dry_run_helper(int depth, int unroll_stage,
                                   int num_iterations)
    {
        LN_LOG(INFO) << tabs.back() << "// DRY_RUN DEPTH: " << depth << "\n";

        std::vector<fma_operation> unrolled_fmas;

        if (depth == unroll_stage)
        {
            unrolled_fmas = collect_unrolled_FMAs_below(depth);
            issue_unrolled_fmas_dry_run(unrolled_fmas, num_iterations);
        }
        else
        {
            auto const& loop = loops[depth];

            std::string var_name = loop.var + "_" + std::to_string(loop.delta);

            auto loop_end        = limits[loop.var].back();
            auto full_iterations = loop_end / loop.delta;
            auto tail            = loop_end % loop.delta;

            if (full_iterations > 0)
            {
                LN_LOG(INFO)
                    << tabs.back() << "FOR: " << var_name << " FROM 0 TO "
                    << loop_end << " BY " << loop.delta << " {\n";
            }

            if (full_iterations > 0)
            {
                limits[loop.var].push_back(loop.delta);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_dry_run_helper(depth + 1, unroll_stage,
                                          num_iterations * full_iterations);
                tabs.pop_back();
                limits[loop.var].pop_back();
            }

            if (full_iterations > 0)
            {
                LN_LOG(INFO) << tabs.back() << "} END FOR\n";
            }

            if (tail)
            {
                LN_LOG(INFO) << tabs.back() << "TAIL: " << var_name << " OF "
                             << tail << " {\n";
                limits[loop.var].push_back(tail);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_dry_run_helper(depth + 1, unroll_stage,
                                          num_iterations);
                tabs.pop_back();
                limits[loop.var].pop_back();
                LN_LOG(INFO) << tabs.back() << "} END TAIL\n";
            }
        }
    }

    void issue_loops_dry_run(int unroll_stage)
    {
        issue_loop_dry_run_helper(0, unroll_stage, 1);
    }

private:
    std::int64_t effective_flops_, masked_out_flops_;
    std::int64_t A_memory_, B_memory_, C_memory_, total_memory_;

    void compute_effective_flops()
    {
        // effective FLOPs are defined as FLOPs that
        // result in values that are actually used (i.e. not masked out
        // operations)
        std::int64_t flops = 2;
        for (auto const& s : sizes)
        {
            if (C_strides.count(s.first) || B_strides.count(s.first) ||
                A_strides.count(s.first))
                flops *= s.second;
        }
        effective_flops_ = flops;
    }

    void compute_masked_out_flops()
    {
        // when the innermost loop (vectorized dimension) doesn't fill up a
        // vector register there are lanes that are masked out, we count the
        // FLOPs resulting from these masked out lanes
        std::pair<std::string, int> innermost = order.back();
        // compute the bound for the innermost loop, since this determines the
        // number of elements vectorized identify bound by looking at stride for
        // prior split (if any)
        auto matches_innermost = [&innermost](auto const& dim) {
            return dim.first == innermost.first;
        };
        auto parent_iter =
            std::find_if(++order.rbegin(), order.rend(), matches_innermost);

        int innermost_bound;
        if (parent_iter == order.rend())
        {
            // there was no split before, so bound is size of dimension
            innermost_bound = sizes.at(innermost.first);
        }
        else
        {
            innermost_bound = parent_iter->second;
        }

        if ((innermost_bound % vector_size) == 0)
        {
            masked_out_flops_ = 0;
            return;
        }

        int masked_out_per_register =
            vector_size - (innermost_bound % vector_size);
        std::string  vectorized_dimension = innermost.first;
        std::int64_t flops                = 2;

        for (auto const& s : sizes)
        {
            if (C_strides.count(s.first) || B_strides.count(s.first) ||
                A_strides.count(s.first))
            {
                if (s.first == vectorized_dimension)
                {
                    flops *=
                        masked_out_per_register * (s.second / innermost_bound);
                }
                else
                {
                    flops *= s.second;
                }
            }
        }
        masked_out_flops_ = flops;
    }

    void compute_memory()
    {
        std::int64_t C_memory_approx = 1;
        std::int64_t A_memory_approx = 1;
        std::int64_t B_memory_approx = 1;

        for (auto const& s : sizes)
        {
            if (C_strides.count(s.first))
                C_memory_approx += (s.second - 1) * C_strides.at(s.first);
            if (A_strides.count(s.first))
                A_memory_approx += (s.second - 1) * A_strides.at(s.first);
            if (B_strides.count(s.first))
                B_memory_approx += (s.second - 1) * B_strides.at(s.first);
        }
        // in bytes
        C_memory_     = C_memory_approx * 4;
        A_memory_     = A_memory_approx * 4;
        B_memory_     = B_memory_approx * 4;
        total_memory_ = C_memory_ + A_memory_ + B_memory_;
    }

public:
    std::int64_t get_effective_flops() const { return effective_flops_; }

    std::int64_t get_masked_out_flops() const { return masked_out_flops_; }

    std::int64_t get_total_memory() const { return total_memory_; }

public:
    FMA_loop_nest_jitter(
        std::vector<std::pair<std::string, int>> const& _order,
        std::map<std::string, int> const&               sizes,
        std::set<std::string> const&                    C_formula,
        std::set<std::string> const&                    A_formula,
        std::set<std::string> const&                    B_formula,
        std::map<std::string, int> const&               C_strides,
        std::map<std::string, int> const&               A_strides,
        std::map<std::string, int> const&               B_strides,
        std::shared_ptr<operation_pair_base>            op_pair,
        std::optional<int> user_fma_unroll_limit = std::nullopt,
        std::shared_ptr<elementwise_operation<ISA>> elementwise_preop = nullptr,
        std::vector<std::map<std::string, int>> const&
                                                    elementwise_preop_strides = {},
        std::shared_ptr<elementwise_operation<ISA>> elementwise_postop =
            nullptr,
        std::vector<std::map<std::string, int>> const&
                                                 elementwise_postop_strides = {},
        std::optional<OptimizationConfiguration> optim_config = std::nullopt)
        : order(_order)
        , sizes(sizes)
        , op_pair(op_pair)
        , elementwise_preop(elementwise_preop)
        , elementwise_postop(elementwise_postop)
        , C_formula(C_formula)
        , A_formula(A_formula)
        , B_formula(B_formula)
        , C_strides(C_strides)
        , A_strides(A_strides)
        , B_strides(B_strides)
        , elementwise_preop_strides(elementwise_preop_strides)
        , elementwise_postop_strides(elementwise_postop_strides)
        , nest_depth(_order.size())
        , max_fmas_unrolled(user_fma_unroll_limit ? *user_fma_unroll_limit
                                                  : default_max_fmas_unrolled)
        , is_C_vectorized(C_strides.count(order.back().first) == 1)
        , is_A_vectorized(A_strides.count(order.back().first) == 1)
        , is_B_vectorized(B_strides.count(order.back().first) == 1)
        , optim_config(optim_config ? *optim_config
                                    : OptimizationConfiguration())
    {
        LN_LOG(DEBUG) << "C is " << (is_C_vectorized ? "" : "NOT ")
                      << "vectorized\n"
                      << "A is " << (is_A_vectorized ? "" : "NOT ")
                      << "vectorized\n"
                      << "B is " << (is_B_vectorized ? "" : "NOT ")
                      << "vectorized\n";
        // At least one tensor has to be vectorized.  Otherwise the
        // innermost loop is over a dummy variable.

        // TODO (nicer error message).
        strong_assert(is_C_vectorized || is_B_vectorized || is_A_vectorized);

        vectorized_var = order.back().first;
        LN_LOG(DEBUG) << "Vectorized along: " << vectorized_var << "\n";

        // compute and set approximate FLOPs and memory
        compute_effective_flops();
        compute_masked_out_flops();
        compute_memory();

        elementwise_strides.insert(elementwise_strides.end(),
                                   elementwise_preop_strides.begin(),
                                   elementwise_preop_strides.end());
        elementwise_strides.insert(elementwise_strides.end(),
                                   elementwise_postop_strides.begin(),
                                   elementwise_postop_strides.end());

        allocate_elementwise_addressing_registers();
        allocate_elementwise_labels();

        set_tensor_traits();
        set_elementwise_tensor_traits();

        set_available_vector_registers();

        set_in_register_tensor_pointers();
        set_in_register_elementwise_tensor_pointers();

        initialize_elementwise_ops();

        int first_loop_that_can_hold_C, unroll_stage,
            total_required_fma_operations;

        std::tie(first_loop_that_can_hold_C, unroll_stage,
                 total_required_fma_operations) = possibly_inject_a_loop();

        initialize_loops_data();

        strong_assert(unroll_stage < loops.size());

        int depth_for_register_blocked_C = first_loop_that_can_hold_C;
        int inner_fma_operations         = total_required_fma_operations;

        if (first_loop_that_can_hold_C < unroll_stage)
        {
            std::tie(depth_for_register_blocked_C, inner_fma_operations) =
                lower_register_blocked_loop(unroll_stage, inner_fma_operations);
        }

        first_unused_vmm_register = assign_vmm_registers(
            first_loop_that_can_hold_C, inner_fma_operations);

        //

        std::vector<fma_operation> unrolled_fmas =
            collect_default_unrolled_FMAs_at(unroll_stage);

        strong_assert(unrolled_fmas.size() == total_required_fma_operations);

        auto addressers = create_addressers(std::move(unrolled_fmas));

        // Regs to be saved: RBX and R12-R15 (we don't use RBP)
        push({r15, r14, r13, r12, rbx});

        if (C_traits.access == SCALAR)
        {
            vzeroall();
            if constexpr (std::is_same_v<ISA, avx512>)
            {
                for (int i = 16; i < 32; ++i)
                {
                    vxorpd(Zmm(i), Zmm(i), Zmm(i));
                }
            }
        }

        if ((A_traits.access == SCALAR && B_traits.access == VECTOR_PACKED) ||
            (A_traits.access == VECTOR_PACKED && B_traits.access == SCALAR) ||
            (A_traits.access == VECTOR_PACKED &&
             B_traits.access == VECTOR_PACKED && C_traits.access != SCALAR))
        {
            issue_loops_dry_run(unroll_stage);
        }

        for (auto const& p : addressers)
        {
            p.second->initialize();
        }

        issue_loops(depth_for_register_blocked_C, unroll_stage, addressers);

        strong_assert(instruction_IRs.size() == 0);

        // for (auto const& lo : load_offsets)
        // {
        //     //std::cout << lo.first << ":";
        //     //for (auto const& o : lo.second)
        //     //    std::cout << " " << o;
        //     //std::cout << "\n";
        // }

        pop({r15, r14, r13, r12, rbx});

        // This is apparently very important as it can slow down
        // legacy SSE code upon return.
        // software.intel.com/en-us/forums/intel-isa-extensions/topic/704023
        vzeroupper();
        ret();

        issue_embedded_constants();
        issue_arithmetic_epilogue();
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook

#endif
