// DEBUG: clang++ -Wall -Wpedantic  -std=c++17 loop_nest.cpp -I./xbyak
// -DCT_ISA=avx2 && ./a.out | grep DIF

// TODO - generalize to FN(alpha C + A * B + W)

// TODO(done) - inject a loop based on the max unroll
// TODO - Better tails (try to reuse the fully unrolled code)

// TODO - remove zero-outs from the tail? (Alpha == 0)

// TODO - document the logic for relu (with the +1 thing) We are using
// the upper 31 bits of AlphaReg store the sum of all visited
// dimensions

#pragma once

#include "address_packer.h"
#include "code_generator.h"
#include "elementwise_operation.h"
#include "isa.h"
#include "log.h"
#include "math.h"
#include "most_frequent_queue.h"
#include "multi_vmm.h"

#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <type_traits>
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

private:
    Reg64 CReg_     = rdi;
    Reg64 AReg_     = rsi;
    Reg64 BReg_     = rdx;
    Reg64 AlphaReg_ = rcx;
    Reg64 loopReg_  = rax;

    Label maskLabel_;

    std::vector<Reg64> addressing_registers = {r8,  r9,  r10, r11,
                                               r13, r14, r15, rbx};

    enum access_kind
    {
        SCALAR,
        VECTOR_PACKED,
        VECTOR_STRIDED
    };

    struct loop_descriptor
    {
        std::string var;
        int         end;
        int         delta;
    };

    struct tensor_traits
    {
        std::string name;
        access_kind access;
        Reg64       reg;
        Label*      stridesLabel;
        int         innermost_stride;
        int         access_len;
    };

    struct memory_argument
    {
        int                  offset;
        tensor_traits const* traits;
        int                  mask;

        // We are not comparing the mask

        bool operator<(memory_argument const& o) const
        {
            return std::tie(offset, traits->name) <
                   std::tie(o.offset, o.traits->name);
        }

        bool operator==(memory_argument const& o) const
        {
            return std::tie(offset, traits->name) ==
                   std::tie(o.offset, o.traits->name);
        }

        std::string readable() const
        {
            assert(traits);
            return traits->name + "[" + std::to_string(offset) + ":" +
                   std::to_string(traits->access == SCALAR ? 1 : vector_size) +
                   "]{" + std::to_string(traits->innermost_stride) + "}";
        }
    };

    struct fma_operation
    {
        memory_argument            dest, src1, src2;
        std::map<std::string, int> coordinates;
    };

    static void print_ld(loop_descriptor const& l)
    {
        LN_LOG(INFO) << "Loop over " << l.var << " from 0 to " << l.end
                     << " by " << l.delta << "\n";
    }

    // This is temporary until I find a better way for nice logging
    // that allows for easy debugging
    std::vector<std::string> tabs = {""};

    struct in_register_tensor_pointer_type
    {
        std::string                name;
        Reg64                      reg;
        std::map<std::string, int> strides;
    };

    std::vector<in_register_tensor_pointer_type> in_register_tensor_pointers;

private:
    // Here we put some default unroll limit.
    static constexpr int default_max_fmas_unrolled = 320;

private:
    std::vector<std::pair<std::string, int>> order;
    std::map<std::string, int> const&        sizes;

    std::shared_ptr<elementwise_operation> elementwise;

    std::set<std::string> const& C_formula;
    std::set<std::string> const& A_formula;
    std::set<std::string> const& B_formula;

    std::map<std::string, int> C_strides;
    std::map<std::string, int> A_strides;
    std::map<std::string, int> B_strides;

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
    Label C_access_strides_label;
    Label A_access_strides_label;
    Label B_access_strides_label;

    // Tensor traits
    tensor_traits C_traits;
    tensor_traits A_traits;
    tensor_traits B_traits;

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

                // TODO(zi) Support for masked tail logic should be here
                for (int i = 0; i < fullIterations; ++i)
                {
                    ret.insert(memory_argument{get_cursor_offset(C_strides),
                                               &C_traits, vector_size});
                    current_coordinate_cursor[loop.var] += vector_size;
                }

                if (rest)
                {
                    ret.insert(memory_argument{get_cursor_offset(C_strides),
                                               &C_traits, rest});
                }

                current_coordinate_cursor[loop.var] = saved_coordinate;
            }
            else
            {
                ret.insert(memory_argument{get_cursor_offset(C_strides),
                                           &C_traits, vector_size});
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
            assert(loop.delta == 1);

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

    // Pushes the pointers (C, A or B) that have strides along the
    // dimension dim.
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

    // Advanced pointers of C, B and A along dimension dim by delta
    // elements.
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
        // much more instruction bytes) byt requires fewer memory
        // accesses.  It probably doens't matter as the reads in the
        // approach above can be pipelined with the writes.

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

            assert(C_traits.access != SCALAR);
        }

        assert(next_vector_register <= auxiliary_registers);

        for (auto const& c : loads)
        {
            LN_LOG(INFO) << tabs.back() << "LOAD " << c.readable() << "\n";

            switch (C_traits.access)
            {
            case SCALAR:
                vxorpd(C_VMMs[c][0], C_VMMs[c][0], C_VMMs[c][0]);
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

            // Set auxiliary horizontal vector regs to zero
            for (int s = 1; s < C_VMMs[c].size(); ++s)
            {
                vxorpd(C_VMMs[c][s], C_VMMs[c][s], C_VMMs[c][s]);
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

        assert(next_vector_register <= auxiliary_registers);

        // issue loads
        for (auto const& c : loads)
        {
            LN_LOG(INFO) << tabs.back() << "LOAD " << c.readable() << "\n";

            switch (C_traits.access)
            {
            case SCALAR:
                vxorpd(C_VMMs[c][0], C_VMMs[c][0], C_VMMs[c][0]);
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

            // Set auxiliary horizontal vector regs to zero
            for (int s = 1; s < C_VMMs[c].size(); ++s)
            {
                vxorpd(C_VMMs[c][s], C_VMMs[c][s], C_VMMs[c][s]);
            }
        }
    }

    void issue_C_loads(std::set<memory_argument> const& loads,
                       bool                             issue_first_alpha_logic)
    {
        std::optional<int> tail_mask;

        // Same code among all ISAs for initializing registers to zero
        for (auto const& c : loads)
        {
            if (c.mask != vector_size)
            {
                assert(!tail_mask || *tail_mask == c.mask);
                tail_mask = c.mask;
            }
        }

        if (issue_first_alpha_logic)
        {
            Label loadDataLabel;
            Label doneInitLabel;

            cmp(AlphaReg_, 0);
            jg(loadDataLabel, T_NEAR);

            // Same code among all ISAs for initializing registers to zero
            for (auto const& c : loads)
            {
                LN_LOG(INFO) << tabs.back() << "ZERO " << c.readable() << "\n";
                for (int s = 0; s < C_VMMs[c].size(); ++s)
                {
                    vxorpd(C_VMMs[c][s], C_VMMs[c][s], C_VMMs[c][s]);
                }
            }

            jmp(doneInitLabel, T_NEAR);
            L(loadDataLabel);

            issue_C_loads(loads, tail_mask);

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

        if (issue_max_alpha_logic && elementwise)
        {
            if (C_traits.access == VECTOR_PACKED ||
                C_traits.access == VECTOR_STRIDED)
            {
                Label not_last_label;
                cmp(AlphaReg_, max_alpha - 1);
                jl(not_last_label, T_NEAR);

                // TODO(zi): Nicer using auxiliary array and extra classes
                elementwise->initialize_vector(this, {Vmm(0)}, R());
                // vxorpd(Vmm(0), Vmm(0), Vmm(0));

                for (auto const& c : stores)
                {
                    C_VMMs[c].reduce(*this);
                    LN_LOG(INFO) << tabs.back() << "RELU " << c.readable()
                                 << " at (" << max_alpha << ")\n";
                    elementwise->process_vector(this, C_VMMs[c][0], {}, R());
                    // vmaxps(C_VMMs[c][0], C_VMMs[c][0], Vmm(0));
                }

                L(not_last_label);
            }
        }

        if (C_traits.access == VECTOR_STRIDED)
        {
            arg_C_strides = Vmm(next_vector_register++);
            vmovups(arg_C_strides, ptr[rip + C_access_strides_label]);
            mov(r12, (1 << vector_size) - 1); // TODO (this is probably already
            kmovw(full_k_mask, r12.cvt32());  // initialized during loads
        }

        assert(next_vector_register <= auxiliary_registers);

        if (tail_mask)
        {
            mov(r12, (1 << (*tail_mask)) - 1);
            kmovw(tail_k_mask, r12.cvt32());

            assert(C_traits.access_len != 1);
        }

        for (auto const& c : stores)
        {
            LN_LOG(INFO) << tabs.back() << "STORE " << c.readable() << "\n";

            C_VMMs[c].reduce(*this);
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
                        vaddps(ymm1, ymm1, Ymm(C_VMMs[c][0].getIdx()));
                    }
                    else
                    {
                        vaddps(zmm1, zmm1, C_VMMs[c][0]);
                    }

                    vextractf128(xmm0, ymm1, 1);
                    vaddps(xmm0, xmm0, xmm1);
                }
                else // avx2_plus
                {
                    vextractf128(xmm0, C_VMMs[c][0], 1);
                    vaddps(xmm0, xmm0, C_VMMs[c][0]);
                }

                // xmm1 = xmm0[1,0]
                vpermilpd(xmm1, xmm0, 1);
                vaddps(xmm0, xmm0, xmm1);
                // xmm1 = xmm0[1,0,3,2]
                vpermilps(xmm1, xmm0, 177);
                vaddps(xmm0, xmm0, xmm1);

                // TODO(zi) BETTER!
                if (issue_max_alpha_logic && elementwise)
                {
                    cmp(AlphaReg_, max_alpha - 1);
                    jl(not_last_label);

                    elementwise->process_scalar(this, xmm0, {xmm1}, R());
                    // xorpd(xmm1, xmm1);
                    // maxps(xmm0, xmm1);

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

        if (issue_max_alpha_logic && elementwise)
        {
            if (C_traits.access == VECTOR_PACKED ||
                C_traits.access == VECTOR_STRIDED)
            {
                Label not_last_label;
                cmp(AlphaReg_, max_alpha - 1);
                jl(not_last_label, T_NEAR);

                // TODO(zi): Nicer using auxiliary array and extra classes
                elementwise->initialize_vector(this, {Vmm(0)}, R());
                // vxorpd(Vmm(0), Vmm(0), Vmm(0));

                for (auto const& c : stores)
                {
                    C_VMMs[c].reduce(*this);
                    LN_LOG(INFO) << tabs.back() << "RELU " << c.readable()
                                 << " at (" << max_alpha << ")\n";
                    elementwise->process_vector(this, C_VMMs[c][0], {}, R());
                    // vmaxps(C_VMMs[c][0], C_VMMs[c][0], Vmm(0));
                }

                L(not_last_label);
            }
        }

        if (tail_mask)
        {
            ymm_tail_mask = Vmm(next_vector_register++);
            vmovups(ymm_tail_mask,
                    ptr[rip + maskLabel_ + 4 * (8 - (*tail_mask))]);
        }

        assert(next_vector_register <= auxiliary_registers);

        for (auto const& c : stores)
        {
            LN_LOG(INFO) << tabs.back() << "STORE " << c.readable() << "\n";

            C_VMMs[c].reduce(*this);

            Label not_last_label;

            switch (C_traits.access)
            {
            case SCALAR:
                vextractf128(xmm0, C_VMMs[c][0], 1);
                vaddps(xmm0, xmm0, C_VMMs[c][0]);

                // xmm1 = xmm0[1,0]
                vpermilpd(xmm1, xmm0, 1);
                vaddps(xmm0, xmm0, xmm1);
                // xmm1 = xmm0[1,0,3,2]
                vpermilps(xmm1, xmm0, 177);
                vaddps(xmm0, xmm0, xmm1);

                // TODO(zi) BETTER!
                if (issue_max_alpha_logic && elementwise)
                {
                    cmp(AlphaReg_, max_alpha - 1);
                    jl(not_last_label);

                    elementwise->process_scalar(this, xmm0, {xmm1}, R());
                    // xorpd(xmm1, xmm1);
                    // maxps(xmm0, xmm1);

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
                assert(!tail_mask || *tail_mask == c.mask);
                tail_mask = c.mask;
            }
        }

        issue_C_stores(stores, tail_mask, max_alpha, issue_max_alpha_logic);
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx512> || std::is_same_v<R, avx2_plus>>
    issue_unrolled_fmas(
        std::vector<fma_operation>                            fmas,
        std::map<int, std::shared_ptr<address_packer>> const& addressers,
        std::optional<int>                                    tail_mask)

    {
        // Assignment of vector registers
        Vmm arg1_register, arg2_register;
        Vmm arg_A_strides, arg_B_strides;

        OpMask tail_k_mask = k2;
        OpMask full_k_mask = k3;
        OpMask temp_k_mask = k4;

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

        assert(next_vector_register <= auxiliary_registers);

        if (A_traits.access == VECTOR_STRIDED ||
            B_traits.access == VECTOR_STRIDED)
        {
            mov(r12, (1 << vector_size) - 1);
            kmovw(full_k_mask, r12.cvt32());
        }

        int mask_size = -1;

        // TODO(zi) issue mask only when required (for avx512 as well)
        auto const ensure_initalized_mask = [&](int mask) {
            assert(tail_mask && "Tail mask was not detected");
            assert(mask == *tail_mask);

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
            assert(is_inside_current_limits(inst.coordinates));
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

        int cycle   = arg1_registers.size();
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
                        // Need to zero out the register
                        // as gather doesn't leaves
                        // previous values
                        vxorpd(arg1_reg, arg1_reg, arg1_reg);
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

            // auto fmas = unrolled_fmas;

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
                        vfmadd231ps(
                            C_VMMs[op.dest]++, arg1_reg,
                            ptr_b[addressers.at(src2.traits->reg.getIdx())
                                      ->get_address(src2.offset * 4)]);
                        break;

                    case VECTOR_PACKED:
                        vfmadd231ps(
                            C_VMMs[op.dest]++, arg1_reg,
                            ptr[addressers.at(src2.traits->reg.getIdx())
                                    ->get_address(src2.offset * 4)]);
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
                        vfmadd231ps(C_VMMs[op.dest]++, arg1_reg, arg2_register);
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

        if (requires_temp_ymm)
        {
            ymm_temp_register = Vmm(next_vector_register++);
        }

        assert(next_vector_register <= auxiliary_registers);

        most_frequent_queue<memory_argument> queue;

        for (auto const& p : addressers)
        {
            p.second->loop_prologue();
        }

        for (auto const& inst : fmas)
        {
            assert(is_inside_current_limits(inst.coordinates));
            queue.inc(inst.src1);
            queue.inc(inst.src2);
        }

        int mask_size = -1;

        // TODO(zi) issue mask only when required (for avx512 as well)
        auto const ensure_initalized_mask = [&](int mask) {
            assert(tail_mask && "Tail mask was not detected");
            assert(mask = *tail_mask);

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

        int cycle   = arg1_registers.size();
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
                    vxorpd(arg2_register, arg2_register, arg2_register);
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
                        vxorpd(arg1_reg, arg1_reg, arg1_reg);
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

            // auto fmas = unrolled_fmas;

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

                        vfmadd231ps(C_VMMs[op.dest]++, arg1_reg, arg2_register);
                        break;

                    case VECTOR_PACKED:
                        vfmadd231ps(
                            C_VMMs[op.dest]++, arg1_reg,
                            ptr[addressers.at(src2.traits->reg.getIdx())
                                    ->get_address(src2.offset * 4)]);
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

                        vfmadd231ps(C_VMMs[op.dest]++, arg1_reg, arg2_register);
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
                assert(!tail_mask || *tail_mask == m);
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

        if (requires_temp_ymm)
        {
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
            assert(it == it_end);
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

            // assert(it_end != order.begin());

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
                inner_fma_operations > 100)
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
            assert(next <= isa_traits<ISA>::total_vector_registers);

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
                    assert((loops[i].end % vector_size) == 0);
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

    std::map<int, std::shared_ptr<address_packer>>
    create_addressers(std::vector<fma_operation> unrolled_fmas)
    {
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
        std::optional<int> user_fma_unroll_limit           = std::nullopt,
        std::shared_ptr<elementwise_operation> elementwise = nullptr)
        : order(_order)
        , sizes(sizes)
        , elementwise(elementwise)
        , C_formula(C_formula)
        , A_formula(A_formula)
        , B_formula(B_formula)
        , C_strides(C_strides)
        , A_strides(A_strides)
        , B_strides(B_strides)
        , nest_depth(_order.size())
        , max_fmas_unrolled(user_fma_unroll_limit ? *user_fma_unroll_limit
                                                  : default_max_fmas_unrolled)
        , is_C_vectorized(C_strides.count(order.back().first) == 1)
        , is_A_vectorized(A_strides.count(order.back().first) == 1)
        , is_B_vectorized(B_strides.count(order.back().first) == 1)
    {
        LN_LOG(DEBUG) << "C is " << (is_C_vectorized ? "" : "NOT ")
                      << "vectorized\n"
                      << "A is " << (is_A_vectorized ? "" : "NOT ")
                      << "vectorized\n"
                      << "B is " << (is_B_vectorized ? "" : "NOT ")
                      << "vectorized\n";
        // At least one tensor has to be vectorized.  Otherwise the
        // innermost loop is over a dummy variable.
        assert(is_C_vectorized || is_B_vectorized || is_A_vectorized);

        vectorized_var = order.back().first;
        LN_LOG(DEBUG) << "Vectorized along: " << vectorized_var << "\n";

        set_tensor_traits();

        set_available_vector_registers();

        set_in_register_tensor_pointers();

        int first_loop_that_can_hold_C, unroll_stage,
            total_required_fma_operations;

        std::tie(first_loop_that_can_hold_C, unroll_stage,
                 total_required_fma_operations) = possibly_inject_a_loop();

        initialize_loops_data();

        assert(unroll_stage < loops.size());

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

        assert(unrolled_fmas.size() == total_required_fma_operations);

        auto addressers = create_addressers(std::move(unrolled_fmas));

        // Regs to be saved: RBX and R12-R15 (we don't use RBP)
        push({r15, r14, r13, r12, rbx});

        for (auto const& p : addressers)
        {
            p.second->initialize();
        }

        issue_loops(depth_for_register_blocked_C, unroll_stage, addressers);

        pop({r15, r14, r13, r12, rbx});

        // This is apparently very important as it can slow down
        // legacy SSE code upon return.
        // software.intel.com/en-us/forums/intel-isa-extensions/topic/704023
        vzeroupper();
        ret();

        issue_embedded_constants();
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
