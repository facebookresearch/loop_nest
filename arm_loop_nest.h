#pragma once

#if defined(LOOP_NEST_ARM)

#include "code_generator.h"
#include "common.h"
#include "isa.h"
#include "log.h"
#include "math.h"
#include "most_frequent_queue.h"
#include "multi_vreg.h"

// #include "address_packer.h"
// #include "elementwise_operation.h"

#include <any>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
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

// https://stackoverflow.com/questions/27941220/push-lr-and-pop-lr-in-arm-arch64

template <class>
class FMA_loop_nest_jitter;

template <>
class FMA_loop_nest_jitter<aarch64>
    : public code_generator<void(float* C, float const* A, float const* B,
                                 int alpha)>
{
private:
    std::vector<std::any> raii;

    using base =
        code_generator<void(float* C, float const* A, float const* B, int)>;
    using Vmm         = VReg;
    using multi_vregs = multi_vreg<Vmm>;

    static constexpr int vector_size = isa_traits<aarch64>::vector_size;

    std::shared_ptr<Label> make_label()
    {
        auto ret = std::make_shared<Label>();
        raii.push_back(ret);
        return ret;
    }

    void meta_push(XReg const& op) { str(op, post_ptr(stackReg_, 8)); }

    void meta_pop(XReg const& op) { ldr(op, pre_ptr(stackReg_, -8)); }

    void meta_push(std::vector<XReg> const& regs)
    {
        for (auto const& r : regs)
        {
            meta_push(r);
        }
    }

    void meta_pop(std::vector<XReg> const& regs)
    {
        for (auto it = regs.crbegin(); it != regs.crend(); ++it)
        {
            meta_pop(*it);
        }
    }

    void prepare_stack()
    {
        // stack_offset = 0;
        sub(sp, sp, 1024);
        sub(sp, sp, 1024);
        mov(stackReg_, sp);
    }

    void restore_stack()
    {
        add(sp, sp, 1024);
        add(sp, sp, 1024);
    }

    template <typename T>
    void mov_imm(const XReg& dst, T imm, const XReg& tmp)
    {
        assert(dst.getIdx() != tmp.getIdx());

        int64_t  bit_ptn = static_cast<int64_t>(imm);
        uint64_t mask    = 0xFFFF;
        bool     flag    = false;

        /* ADD(immediate) supports unsigned imm12 */
        const uint64_t IMM12_MASK = ~uint64_t(0xfff);
        if ((bit_ptn & IMM12_MASK) == 0)
        { // <= 4095
            mov(dst, static_cast<uint32_t>(imm & 0xfff));
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
                    movz(dst, static_cast<uint32_t>(tmp_ptn), i);
                    flag = true;
                }
                else
                {
                    movz(tmp, static_cast<uint32_t>(tmp_ptn), i);
                    add(dst, dst, tmp);
                }
            }
        }

        return;
    }

    template <typename T>
    void mov_imm(const XReg& dst, T imm)
    {
        mov_imm(dst, imm, xtmp1);
    }

private:
    Reg64 CReg_     = x0;
    Reg64 AReg_     = x1;
    Reg64 BReg_     = x2;
    Reg64 AlphaReg_ = x3;
    Reg64 ZeroReg_  = x4;
    Reg64 xtmp1     = x5;
    Reg64 xtmp2     = x6;
    Reg64 loopReg_  = x7;
    Reg64 stackReg_ = x9;
    Reg64 tmpCReg_  = x10;
    Reg64 tmpAReg_  = x11;
    Reg64 tmpBReg_  = x12;
    // std::vector<Reg64> elementwiseReg_;
    std::map<int, Reg64> const_regs;

    VReg ZeroVector_ = v0;

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

    // std::shared_ptr<elementwise_operation<aarch64>> elementwise_preop;
    // std::shared_ptr<elementwise_operation<aarch64>> elementwise_postop;

    std::set<std::string> const& C_formula;
    std::set<std::string> const& A_formula;
    std::set<std::string> const& B_formula;

    std::map<std::string, int> C_strides;
    std::map<std::string, int> A_strides;
    std::map<std::string, int> B_strides;
    // std::vector<std::map<std::string, int>> elementwise_preop_strides;
    // std::vector<std::map<std::string, int>> elementwise_postop_strides;
    // concatenation of preop followed by postop strides
    // std::vector<std::map<std::string, int>> elementwise_strides;

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
    // std::vector<std::shared_ptr<Label>> elementwise_labels;

    // Tensor traits
    tensor_traits C_traits;
    tensor_traits A_traits;
    tensor_traits B_traits;
    // std::vector<tensor_traits> elementwise_traits;

    // The name of the variable in the innermost loop (along which the
    // vectorization is performed)
    std::string vectorized_var;

    // Assignment of registers for register blocking of the values of C
    std::map<memory_argument, multi_vregs> C_VMMs;

    // Number of auxillary registers (used for pre-loading and bradcasting, as
    // well as horizontal add at the end)
    int auxiliary_registers;

    // Number of available vector registers for computing
    int available_registers;

    // First register that will not be used by C or auxiliary
    // registers.  Can be used for software pipelining.  Set to
    // isa_traits<aarch64>::total_vector_registers if none available
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
    void allocate_elementwise_addressing_registers() {}

    void allocate_elementwise_labels() {}

    void initialize_elementwise_ops() {}

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

    template <class T>
    void add_imm(XReg const& srcdst, T imm)
    {
        if (imm == 0)
            return;

        base::add_imm(srcdst, srcdst, imm, xtmp1);
    }

    template <class T>
    void sub_imm(XReg const& srcdst, T imm)
    {
        if (imm == 0)
            return;

        base::sub_imm(srcdst, srcdst, imm, xtmp1);
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
                LN_LOG(INFO) << tabs.back() << "PUSH " << ptr.name << "(X"
                             << ptr.reg.getIdx() << ")\n";
                meta_push(ptr.reg);
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
                LN_LOG(INFO) << tabs.back() << "POP " << ptr.name << "(X"
                             << ptr.reg.getIdx() << ")\n";
                meta_pop(ptr.reg);
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
                    << tabs.back() << ptr.name << "(X" << ptr.reg.getIdx()
                    << ") += " << delta << " * " << ptr.strides.at(dim) << "\n";
                add_imm(ptr.reg, ptr.strides.at(dim) * delta * 4);
            }
        }
    };

    void load_scalar(VReg const& vreg, Reg64 const& base, int offset,
                     int increment = 0)
    {
        if (offset)
        {
            add_imm(base, offset);
        }

        mov(vreg.b16, ZeroVector_.b16);
        if (increment && increment < 256)
        {
            ldr(SReg(vreg.s4.getIdx()), post_ptr(base, increment));
            increment = 0;
        }
        else
        {
            ldr(SReg(vreg.s4.getIdx()), ptr(base));
        }

        if (offset)
        {
            sub_imm(base, offset);
        }
        if (increment)
        {
            add_imm(base, increment);
        }
    }

    void store_scalar(VReg const& vreg, Reg64 const& base, int offset,
                      int increment = 0)
    {
        if (offset)
        {
            add_imm(base, offset);
        }

        if (increment && increment < 256)
        {
            str(SReg(vreg.s4.getIdx()), post_ptr(base, increment));
            increment = 0;
        }
        else
        {
            str(SReg(vreg.s4.getIdx()), ptr(base));
        }

        if (offset)
        {
            sub_imm(base, offset);
        }
        if (increment)
        {
            add_imm(base, increment);
        }
    }

    void broadcast_scalar(VReg const& vreg, Reg64 const& base, int offset,
                          int mask = vector_size, int increment = 0)
    {
        if (offset)
        {
            add_imm(base, offset);
        }

        if (mask == 4)
        {
            ld1r(vreg.s4, ptr(base));
        }
        else
        {
            for (int i = 0; i < mask; ++i)
            {
                ld1(vreg.s[i], ptr(base));
            }
        }

        if (offset)
        {
            sub_imm(base, offset);
        }
        if (increment)
        {
            add_imm(base, increment);
        }
    }

    void load_vector(VReg const& vreg, Reg64 const& base, int offset, int mask,
                     int increment = 0)
    {
        if (offset)
        {
            add_imm(base, offset);
        }

        if (mask == vector_size || mask == 3)
        {
            if (increment && increment < 256)
            {
                ldr(QReg(vreg.s4.getIdx()), post_ptr(base, increment));
                increment = 0;
            }
            else
            {
                ldr(QReg(vreg.s4.getIdx()), ptr(base));
            }
        }
        else if (mask == 2)
        {
            ldr(DReg(vreg.s4.getIdx()), ptr(base));
        }
        else if (mask == 1)
        {
            ldr(SReg(vreg.s4.getIdx()), ptr(base));
        }
        else
        {
            mov(vreg.b16, ZeroVector_.b16);
            ld1(vreg.s4[0], ptr(base));
            for (int i = 1; i < mask; ++i)
            {
                add_imm(base, 4);
                offset += 4;
                ld1(vreg.s4[i], ptr(base));
            }
        }

        if (offset)
        {
            sub_imm(base, offset);
        }
        if (increment)
        {
            add_imm(base, increment);
        }
    }

    void store_vector(VReg const& vreg, Reg64 const& base, int offset, int mask,
                      int increment = 0)
    {
        if (offset)
        {
            add_imm(base, offset);
        }

        if (mask == vector_size)
        {
            if (increment && increment < 256)
            {
                str(QReg(vreg.s4.getIdx()), post_ptr(base, increment));
                increment = 0;
            }
            else
            {
                str(QReg(vreg.s4.getIdx()), ptr(base));
            }
        }
        else
        {
            st1(vreg.s4[0], ptr(base));
            for (int i = 1; i < mask; ++i)
            {
                add_imm(base, 4);
                offset += 4;
                st1(vreg.s4[i], ptr(base));
            }
        }

        if (offset)
        {
            sub_imm(base, offset);
        }
        if (increment)
        {
            add_imm(base, increment);
        }
    }

    void scatter_vector(VReg const& vreg, Reg64 const& base, int offset,
                        int mask, int stride, int increment = 0)
    {
        if (mask == 0)
        {
            return;
        }

        if (offset)
        {
            add_imm(base, offset);
        }

        st1(vreg.s4[0], ptr(base));

        for (int i = 1; i < mask; ++i)
        {
            add_imm(base, stride);
            offset += stride;
            st1(vreg.s4[i], ptr(base));
        }

        if (offset)
        {
            sub_imm(base, offset);
        }
        if (increment)
        {
            add_imm(base, increment);
        }
    }

    void gather_vector(VReg const& vreg, Reg64 const& base, int offset,
                       int mask, int stride, int increment = 0)
    {
        if (mask == 0)
        {
            return;
        }

        if (offset)
        {
            add_imm(base, offset);
        }

        ld1(vreg.s4[0], ptr(base));

        for (int i = 1; i < mask; ++i)
        {
            add_imm(base, stride);
            offset += stride;
            ld1(vreg.s4[i], ptr(base));
        }

        if (offset)
        {
            sub_imm(base, offset);
        }
        if (increment)
        {
            add_imm(base, increment);
        }
    }

    void issue_C_loads(std::set<memory_argument> const& loads)
    {
        std::vector<memory_argument> ordered_loads;
        for (auto const& c : loads)
        {
            ordered_loads.emplace_back(c);
        }
        std::sort(ordered_loads.begin(), ordered_loads.end(),
                  [](const memory_argument& a, const memory_argument& b) {
                      return a.offset < b.offset;
                  });
        std::vector<int> incrs;
        int              prev_off = -1;
        for (auto const& c : ordered_loads)
        {
            if (prev_off != -1)
            {
                incrs.emplace_back(c.offset - prev_off);
            }
            prev_off = c.offset;
        }
        mov(tmpCReg_, CReg_);
        assert(ordered_loads.size());
        add_imm(tmpCReg_, ordered_loads.front().offset * 4);
        for (auto const& c : ordered_loads)
        {
            LN_LOG(INFO) << tabs.back() << "LOAD " << c.readable() << "\n";

            // Move the reg pointer
            auto incr = 0;
            if (incrs.size())
            {
                incr = incrs.front() * 4;
                incrs.erase(incrs.begin());
            }

            switch (C_traits.access)
            {
            case SCALAR:
                load_scalar(C_VMMs[c][0], tmpCReg_, 0, incr);
                break;

            case VECTOR_PACKED:
                load_vector(C_VMMs[c][0], tmpCReg_, 0, c.mask, incr);
                break;

            case VECTOR_STRIDED:
                gather_vector(C_VMMs[c][0], tmpCReg_, 0, c.mask,
                              C_traits.innermost_stride * 4, incr);
                break;
            }

            // Set auxiliary horizontal vector regs to zero
            for (int s = 1; s < C_VMMs[c].size(); ++s)
            {
                mov(C_VMMs[c][s].b16, ZeroVector_.b16);
            }
        }
    }

    void issue_C_elementwise_preop(std::set<memory_argument> const& loads)
    {
        switch (C_traits.access)
        {
        case VECTOR_STRIDED:
            /* fall through to vector packed case */
        case VECTOR_PACKED:
            break;

        case SCALAR:
            break;
        }
    }

    void issue_C_loads(std::set<memory_argument> const& loads,
                       bool                             issue_first_alpha_logic)
    {
        if (issue_first_alpha_logic)
        {
            auto loadDataLabel = make_label();
            auto doneInitLabel = make_label();

            cbnz(AlphaReg_, *loadDataLabel);

            for (auto const& c : loads)
            {
                LN_LOG(INFO) << tabs.back() << "ZERO " << c.readable() << "\n";
                for (int s = 0; s < C_VMMs[c].size(); ++s)
                {
                    mov(C_VMMs[c][s].b16, ZeroVector_.b16);
                }
            }

            b(*doneInitLabel);

            L_aarch64(*loadDataLabel);
            issue_C_loads(loads);

            L_aarch64(*doneInitLabel);
        }
        else
        {
            issue_C_loads(loads);
        }
    }

    void issue_C_stores(std::set<memory_argument> const& stores,
                        std::optional<int> tail_mask, int max_alpha,
                        bool issue_max_alpha_logic)
    {
        std::vector<memory_argument> ordered_stores;
        for (auto const& c : stores)
        {
            ordered_stores.emplace_back(c);
        }
        std::sort(ordered_stores.begin(), ordered_stores.end(),
                  [](const memory_argument& a, const memory_argument& b) {
                      return a.offset < b.offset;
                  });
        std::vector<int> incrs;
        int              prev_off = -1;
        for (auto const& c : ordered_stores)
        {
            if (prev_off != -1)
            {
                incrs.emplace_back(c.offset - prev_off);
            }
            prev_off = c.offset;
        }
        mov(tmpCReg_, CReg_);
        assert(ordered_stores.size());
        add_imm(tmpCReg_, ordered_stores.front().offset * 4);
        for (auto const& c : ordered_stores)
        {
            LN_LOG(INFO) << tabs.back() << "STORE " << c.readable() << "\n";

            auto incr = 0;
            if (incrs.size())
            {
                incr = incrs.front() * 4;
                incrs.erase(incrs.begin());
            }

            C_VMMs[c].reduce(*this);

            switch (C_traits.access)
            {
            case SCALAR:
                C_VMMs[c].full_reduce(*this, c.mask);
                store_scalar(C_VMMs[c][0], tmpCReg_, 0, incr);
                break;

            case VECTOR_PACKED:
                store_vector(C_VMMs[c][0], tmpCReg_, 0, c.mask, incr);
                break;

            case VECTOR_STRIDED:
                scatter_vector(C_VMMs[c][0], tmpCReg_, 0, c.mask,
                               C_traits.innermost_stride * 4, incr);
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

    void issue_unrolled_fmas_scalar_vector(std::vector<fma_operation> fmas)
    {
        std::cout << "ZI WAS HERE\n";
    }

    void issue_unrolled_fmas(std::vector<fma_operation> fmas)
    {
        if (0 && fmas.size())
        {
            if (fmas[0].src1.traits->access == SCALAR &&
                fmas[0].src2.traits->access == VECTOR_PACKED)
            {
                issue_unrolled_fmas_scalar_vector(std::move(fmas));
                return;
            }
            else if (fmas[0].src1.traits->access == VECTOR_PACKED &&
                     fmas[0].src2.traits->access == SCALAR)
            {
                for (auto& f : fmas)
                {
                    std::swap(f.src1, f.src2);
                }
                issue_unrolled_fmas_scalar_vector(std::move(fmas));
                return;
            }
        }

        most_frequent_queue<memory_argument> queue;

        // coalesce broadcasting loads
        auto normalize_broadcast = [](memory_argument m) {
            if (m.traits->access == SCALAR)
            {
                auto new_off = vector_size * (m.offset / vector_size);
                return memory_argument{new_off, m.traits, vector_size,
                                       m.coordinates};
            }
            return m;
        };

        std::set<memory_argument> can_implicit_broadcast;
        for (auto const& inst : fmas)
        {
            assert(is_inside_current_limits(inst.coordinates));
            queue.inc(inst.src1);
            auto src2 = normalize_broadcast(inst.src2);
            queue.inc(src2);
            can_implicit_broadcast.insert(src2);
        }

        auto num_regs = isa_traits<aarch64>::total_vector_registers -
                        first_unused_vmm_register;

        std::set<int> free_regs;
        for (auto i = 0; i < num_regs; ++i)
        {
            free_regs.insert(first_unused_vmm_register + i);
        }
        std::map<memory_argument, int> vmm_map;

        while (queue.size())
        {

            std::vector<memory_argument> loads;
            for (; free_regs.size() && queue.size();)
            {
                auto addr = queue.top();
                if (!vmm_map.count(addr))
                {
                    vmm_map[addr] = *free_regs.begin();
                    free_regs.erase(vmm_map.at(addr));
                    loads.emplace_back(addr);
                }
                queue.pop();
            }

            // reduce loads for uncoalesced broadcasts
            std::map<memory_argument, int> load_mask;
            for (auto const& fma : fmas)
            {
                // keep track of needed loads for broadcasting
                if (fma.src2.traits->access == SCALAR)
                {
                    auto idx  = fma.src2.offset % vector_size;
                    auto src2 = normalize_broadcast(fma.src2);
                    load_mask[normalize_broadcast(fma.src2)] =
                        std::max(load_mask[src2], idx);
                }
            }

            std::vector<fma_operation> in_vmm_fmas;
            for (auto it = fmas.begin(); it != fmas.end();)
            {
                auto src1 = it->src1;
                auto src2 = normalize_broadcast(it->src2);
                if (vmm_map.count(src1) && vmm_map.count(src2))
                {
                    in_vmm_fmas.emplace_back(*it);
                    it = fmas.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            if (!in_vmm_fmas.size())
            {
                issue_unrolled_fmas_(fmas);
                return;
            }

            std::sort(loads.begin(), loads.end(),
                      [](const memory_argument& a, const memory_argument& b) {
                          return a.offset < b.offset;
                      });

            std::map<int, std::pair<int, int>> tmp_addresser;
            std::map<int, int>                 tmp_addresser_base;
            std::map<int, std::vector<int>>    load_incrs;
            for (auto const& addr : loads)
            {
                auto base_reg = addr.traits->reg.getIdx();
                if (!tmp_addresser.count(base_reg))
                {
                    int tmp = base_reg == BReg_.getIdx() ? tmpBReg_.getIdx()
                                                         : tmpAReg_.getIdx();
                    tmp_addresser[base_reg] = std::make_pair(tmp, addr.offset);
                    tmp_addresser_base[tmp] = addr.offset;
                    load_incrs[base_reg];
                }
                else
                {
                    auto prev_off = tmp_addresser[base_reg].second;
                    load_incrs[base_reg].emplace_back(addr.offset - prev_off);
                    tmp_addresser[base_reg].second = addr.offset;
                }
            }
            std::map<memory_argument, int> post_ptr_map;
            for (auto const& addr : loads)
            {
                auto base_reg = addr.traits->reg.getIdx();
                if (load_incrs.at(base_reg).size())
                {
                    post_ptr_map[addr] = load_incrs.at(base_reg).front();
                    load_incrs.at(base_reg).erase(
                        load_incrs.at(base_reg).begin());
                }
            }

            for (auto const& kv : tmp_addresser)
            {
                auto orig_reg  = kv.first;
                auto tmp_reg   = kv.second.first;
                auto orig_base = tmp_addresser_base.at(tmp_reg);
                mov(Reg64(tmp_reg), Reg64(orig_reg));
                add_imm(Reg64(tmp_reg), orig_base * 4);
            }

            for (auto const& addr : loads)
            {

                auto reg = Vmm(vmm_map.at(addr));
                auto addr_reg =
                    Reg64(tmp_addresser.at(addr.traits->reg.getIdx()).first);
                auto inc = 0;
                if (post_ptr_map.count(addr))
                {
                    inc = post_ptr_map.at(addr) * 4;
                }
                switch (addr.traits->access)
                {
                case SCALAR:
                    if (can_implicit_broadcast.count(addr))
                    {
                        load_vector(reg, addr_reg, 0,
                                    load_mask.count(addr)
                                        ? load_mask.at(addr) + 1
                                        : vector_size,
                                    inc);
                    }
                    else
                    {
                        broadcast_scalar(reg, addr_reg, 0, addr.mask, inc);
                    }
                    break;
                case VECTOR_PACKED:
                    load_vector(reg, addr_reg, 0, vector_size, inc);
                    break;

                case VECTOR_STRIDED:
                    gather_vector(reg, addr_reg, 0, addr.mask,
                                  addr.traits->innermost_stride * 4, inc);
                    break;
                }
            }

            // emit fmas
            for (auto const& fma : in_vmm_fmas)
            {
                auto arg1_reg = Vmm(vmm_map.at(fma.src1));
                auto arg2_reg = Vmm(vmm_map.at(normalize_broadcast(fma.src2)));
                if (fma.src2.traits->access == SCALAR)
                {
                    auto idx = fma.src2.offset % vector_size;
                    fmla((C_VMMs[fma.dest]++).s4, arg1_reg.s4,
                         arg2_reg.s4[idx]);
                }
                else
                {
                    fmla((C_VMMs[fma.dest]++).s4, arg1_reg.s4, arg2_reg.s4);
                }
            }

            queue = most_frequent_queue<memory_argument>();
            std::set<memory_argument> keep;
            for (auto const& inst : fmas)
            {
                auto src2 = normalize_broadcast(inst.src2);
                queue.inc(inst.src1);
                if (vmm_map.count(src2))
                {
                    if (src2.traits->access == SCALAR)
                    {
                        keep.insert(src2);
                    }
                    else
                    {
                        queue.inc(normalize_broadcast(inst.src2));
                    }
                }
                else
                {
                    queue.inc(normalize_broadcast(inst.src2));
                }
            }
            std::vector<memory_argument> to_free;
            for (auto const& kv : vmm_map)
            {
                if (keep.count(kv.first))
                {
                    continue;
                }
                to_free.emplace_back(kv.first);
                free_regs.insert(kv.second);
            }
            for (auto const& m : to_free)
            {
                vmm_map.erase(m);
            }

        } // while queue.size()
    }

    void issue_unrolled_fmas_(std::vector<fma_operation> fmas)
    {
        most_frequent_queue<memory_argument> queue;

        for (auto const& inst : fmas)
        {
            // Ensures no instructions are added to the unrolled
            // loop tails
            assert(is_inside_current_limits(inst.coordinates));
            queue.inc(inst.src1);
            queue.inc(inst.src2);
        }

        Vmm arg1_register = Vmm(1);
        Vmm arg2_register = Vmm(2);

        std::vector<Vmm> arg1_registers;
        arg1_registers.push_back(arg1_register);
        for (int i = first_unused_vmm_register;
             i < isa_traits<aarch64>::total_vector_registers; ++i)
        {
            arg1_registers.push_back(Vmm(i));
        }

        // TODO(zi) replace this eyeballed value
        while (arg1_registers.size() > 5)
        {
            arg1_registers.pop_back();
            // arg1_registers.resize(5);
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
                broadcast_scalar(arg1_reg, addr.traits->reg, addr.offset * 4,
                                 vector_size);
                break;

            case VECTOR_PACKED:
                load_vector(arg1_reg, addr.traits->reg, addr.offset * 4,
                            C_traits.access == SCALAR ? addr.mask
                                                      : vector_size);
                break;

            case VECTOR_STRIDED:
                gather_vector(arg1_reg, addr.traits->reg, addr.offset * 4,
                              addr.mask, addr.traits->innermost_stride * 4);
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
                                                  delayed_fma_operations, this,
                                                  arg2_register]() {
                for (auto const& op : delayed_fma_operations)
                {
                    auto src1 = op.src1;
                    auto src2 = op.src2;

                    if (addr == src2)
                    {
                        std::swap(src1, src2);
                    }

                    switch (src2.traits->access)
                    {
                    case SCALAR:
                        load_scalar(arg2_register, src2.traits->reg,
                                    src2.offset * 4);
                        break;

                    case VECTOR_PACKED:
                        load_vector(arg2_register, src2.traits->reg,
                                    src2.offset * 4, vector_size);
                        break;

                    case VECTOR_STRIDED:
                        gather_vector(arg2_register, src2.traits->reg,
                                      src2.offset * 4, src2.mask,
                                      src2.traits->innermost_stride * 4);
                        break;
                    }
                    if (src2.traits->access == SCALAR)
                    {
                        fmla((C_VMMs[op.dest]++).s4, arg1_reg.s4,
                             arg2_register.s4[0]);
                    }
                    else
                    {
                        fmla((C_VMMs[op.dest]++).s4, arg1_reg.s4,
                             arg2_register.s4);
                    }
                }
            };
        }

        for (int off = 0; off < cycle; ++off, ++current)
        {
            issue_delayed_ops[current % cycle]();
        }

        // for (auto const& p : addressers)
        // {
        //     p.second->restore();
        // }
    }

    void issue_embedded_constants() {}

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

    void set_elementwise_tensor_traits() {}

    void set_available_vector_registers()
    {
        auxiliary_registers = 3;

        available_registers =
            isa_traits<aarch64>::total_vector_registers - auxiliary_registers;

        LN_LOG(DEBUG) << "AVAILABLE REGS: " << available_registers << "\n";
    }

    void set_in_register_tensor_pointers()
    {
        in_register_tensor_pointers.push_back({"A_Tensor", AReg_, A_strides});
        in_register_tensor_pointers.push_back({"B_Tensor", BReg_, B_strides});
        in_register_tensor_pointers.push_back({"C_Tensor", CReg_, C_strides});
    }

    void set_in_register_elementwise_tensor_pointers() {}

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

            // TODO(ZI) REMOVE
            per_register = 1;

            for (auto const& c : collected_load_store)
            {
                LN_LOG(DEBUG) << "LOAD/STORE: " << c.readable() << " ("
                              << per_register << " VMMs)\n";
                C_VMMs[c] = multi_vregs(per_register, next);
                next += per_register;
            }
            assert(next <= isa_traits<aarch64>::total_vector_registers);

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

    void issue_loop_helper(int depth, bool save_loop, bool save_ptrs,
                           int depth_for_register_blocked_C, int unroll_stage,
                           bool issue_first_alpha_logic, int max_alpha,
                           bool issue_max_alpha_logic)
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
            issue_unrolled_fmas(unrolled_fmas);
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
                meta_push(loopReg_);
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
                mov_imm(loopReg_, full_iterations);
                auto loopLabel = make_label();
                L_aarch64(*loopLabel);

                // --------------------------------------------------
                // RECURSION
                if (depth < depth_for_register_blocked_C &&
                    C_formula.count(loop.var) == 0)
                {
                    new_max_alpha += (full_iterations - 1 + (tail ? 1 : 0)) * 2;
                }

                limits[loop.var].push_back(loop.delta);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_helper(depth + 1, true, true,
                                  depth_for_register_blocked_C, unroll_stage,
                                  issue_first_alpha_logic, new_max_alpha,
                                  recursive_issue_max_alpha_logic);
                tabs.pop_back();
                limits[loop.var].pop_back();
                // --------------------------------------------------
                // RECURSION

                advance_pointers(loop.var, loop.delta);

                if (depth < depth_for_register_blocked_C &&
                    C_formula.count(loop.var) == 0)
                {
                    add_imm(AlphaReg_, 2);
                }

                Label doneLabel;

                sub_imm(loopReg_, 1);
                cbnz(loopReg_, *loopLabel);

                // adr(x15, loopLabel);
                // br(x15);
                // L_aarch64(doneLabel);
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
                                  issue_first_alpha_logic, new_max_alpha,
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
                        add_imm(AlphaReg_, 2);
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
                    depth_for_register_blocked_C, unroll_stage,
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
                sub_imm(AlphaReg_, full_iterations * 2);
            }

            if (full_iterations > 1 && save_loop)
            {
                meta_pop(loopReg_);
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

    void issue_loops(int depth_for_register_blocked_C, int unroll_stage)
    {
        issue_loop_helper(0, false, false, depth_for_register_blocked_C,
                          unroll_stage, true, 1, true);
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
        std::optional<int> user_fma_unroll_limit = std::nullopt)
        : order(_order)
        , sizes(sizes)
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

        // TODO (nicer error message).
        assert(is_C_vectorized || is_B_vectorized || is_A_vectorized);

        vectorized_var = order.back().first;
        LN_LOG(DEBUG) << "Vectorized along: " << vectorized_var << "\n";

        // compute and set approximate FLOPs and memory
        compute_effective_flops();
        compute_masked_out_flops();
        compute_memory();

        // elementwise_strides.insert(elementwise_strides.end(),
        //                            elementwise_preop_strides.begin(),
        //                            elementwise_preop_strides.end());
        // elementwise_strides.insert(elementwise_strides.end(),
        //                            elementwise_postop_strides.begin(),
        //                            elementwise_postop_strides.end());

        // allocate_elementwise_addressing_registers();
        // allocate_elementwise_labels();

        set_tensor_traits();
        // set_elementwise_tensor_traits();

        set_available_vector_registers();

        set_in_register_tensor_pointers();
        // set_in_register_elementwise_tensor_pointers();

        // initialize_elementwise_ops();

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

        // Regs to be saved: RBX and R12-R15 (we don't use RBP)
        // push({r15, r14, r13, r12, rbx});

        prepare_stack();
        eor(ZeroReg_, ZeroReg_, ZeroReg_);
        ins(ZeroVector_.d[0], ZeroReg_);
        ins(ZeroVector_.d[1], ZeroReg_);

        issue_loops(depth_for_register_blocked_C, unroll_stage);

        restore_stack();

        // pop({r15, r14, r13, r12, rbx});

        // This is apparently very important as it can slow down
        // legacy SSE code upon return.
        // software.intel.com/en-us/forums/intel-isa-extensions/topic/704023
        // vzeroupper();
        ret();

        // issue_embedded_constants();
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook

#else
#include "loop_nest.h"
#endif
