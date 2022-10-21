// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/isa.hpp"
#ifdef DABUN_ARCH_X86_64

#    include "dabun/code_generator/code_generator.hpp"
#    include "dabun/common.hpp"
#    include "dabun/math.hpp"
#    include "dabun/utility/log.hpp"

#    include <cassert>
#    include <iostream>
#    include <map>
#    include <numeric>
#    include <optional>
#    include <string>
#    include <type_traits>
#    include <utility>
#    include <vector>

namespace dabun
{
namespace x86
{

template <class ISA>
class transposer_code_generator
    : public basic_code_generator,
      public with_signature<transposer_code_generator<ISA>,
                            void(float* Out, float const* In)>
{
private:
    static constexpr int vector_size = isa_traits<ISA>::vector_size;

    using mask_register_type =
        std::conditional_t<std::is_same_v<ISA, avx2>, Ymm, OpMask>;

    using Vmm = std::conditional_t<std::is_same_v<ISA, avx512>, Zmm, Ymm>;

    // Here we put some default unroll limit.
    static constexpr int default_max_unrolled_moves = 32;

private:
    Reg64 out_reg  = rdi;
    Reg64 in_reg   = rsi;
    Reg64 loop_reg = rax; // TODO(zi) Use multiple loop registers for efficiency

    Label mask_label;

    std::vector<std::pair<std::string, int>> order;
    std::map<std::string, int>               sizes;
    std::map<std::string, int>               out_strides;
    std::map<std::string, int>               in_strides;

    std::string vectorized_var;

    int nest_depth;

    // Used vector (or mask registers). Here the avx512 and avx2_plus
    // will have the coorect k0 initialized We have more than enough
    // vector registers, so that we waste some for no particular
    // reason - just to make the code easier - avx2 case will use real
    // vector registers, while avx512 uses Opmask registers, and the
    // corresponding vector registers are not used.  Register renaming
    // will do it's magic on the die.
    mask_register_type full_mask_reg     = mask_register_type(1);
    mask_register_type tail_mask_reg     = mask_register_type(2);
    mask_register_type in_temp_mask_reg  = mask_register_type(3);
    mask_register_type out_temp_mask_reg = mask_register_type(4);

    // Possible regs that store the gather/scatter strides
    Vmm in_access_strides_reg  = Vmm(5);
    Vmm out_access_strides_reg = Vmm(6);

    // In/out data through the vector reg
    Vmm in_out_vmm_reg = Vmm(0);

    int max_unrolled_moves;

    // This is temporary until I find a better way for nice logging
    // that allows for easy debugging
    std::vector<std::string> tabs = {""};

    void check_representation()
    {
        // Make sure strides (and sizes) exist for each order variable
        for ([[maybe_unused]] auto const& o : order)
        {
            assert(out_strides.count(o.first) > 0);
            assert(in_strides.count(o.first) > 0);
            assert(sizes.count(o.first) > 0);
        }

        std::map<std::string, int> last_step;

        for (auto const& o : order)
        {
            if (last_step.count(o.first))
            {
                assert(last_step[o.first] >= o.second &&
                       "The steps in 'order' need to be non-increasing");
            }
            last_step[o.first] = o.second;
        }

        for ([[maybe_unused]] auto const& o : order)
        {
            assert(last_step[o.first] == 1 &&
                   "Last step in order not equal to 1");
        }

        for (auto const& o : order)
        {
            if (o.first == vectorized_var)
            {
                // std::cout << "VEK VAR: " << vectorized_var
                //           << " ----------------- " << o.second << "\n";
                assert(o.second == 1 || (o.second % vector_size == 0));
            }
        }
    }

    using memory_argument = memory_argument_type<vector_size>;

    std::vector<in_register_tensor_pointer_type> in_register_tensor_pointers;

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

    struct move_operation
    {
        memory_argument dest, src;
    };

    // Tensor traits
    tensor_traits out_traits;
    tensor_traits in_traits;

    // Labels holding strides along LSD of vectorized tensors that are
    // not packed.
    Label in_access_strides_label;
    Label out_access_strides_label;

    void set_tensor_traits()
    {
        // Strides along the LSD dimension of the compute order
        int in_access_stride  = in_strides.at(order.back().first);
        int out_access_stride = out_strides.at(order.back().first);

        LN_LOG(DEBUG) << "in access stride: " << in_access_stride << "\n";
        LN_LOG(DEBUG) << "out access stride: " << out_access_stride << "\n";

        access_kind in_access_kind =
            in_access_stride != 1 ? VECTOR_STRIDED : VECTOR_PACKED;
        access_kind out_access_kind =
            out_access_stride != 1 ? VECTOR_STRIDED : VECTOR_PACKED;

        // TODO remove redundant information.
        in_traits  = {"in",
                     in_access_kind,
                     in_reg,
                     &in_access_strides_label,
                     in_access_stride,
                     vector_size};
        out_traits = {"out",
                      out_access_kind,
                      out_reg,
                      &out_access_strides_label,
                      out_access_stride,
                      vector_size};
    }

    void scatter_avx2_register(Ymm ymm, int mask, Xbyak::RegExp const& base,
                               int stride)
    {
        vmovups(ptr[rsp - 32], ymm);
        for (int i = 0; i < mask; ++i)
        {
            mov(rcx.cvt32(), dword[rsp - 32 + i * 4]);
            mov(dword[base + i * stride], rcx.cvt32());
        }
    }

    void issue_embedded_constants()
    {
        align_to(4);

        if (in_traits.access == VECTOR_STRIDED)
        {
            L(in_access_strides_label);
            for (int i = 0; i < vector_size; ++i)
            {
                dd(i * in_traits.innermost_stride * 4 /* bytes */);
            }
        }
        if (out_traits.access == VECTOR_STRIDED)
        {
            L(out_access_strides_label);
            for (int i = 0; i < vector_size; ++i)
            {
                dd(i * out_traits.innermost_stride * 4 /* bytes */);
            }
        }

        // Will be used as a mask for AVX2
        if (std::is_same_v<ISA, avx2>)
        {
            L(mask_label);
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

    void set_in_register_tensor_pointers()
    {
        in_register_tensor_pointers.push_back(
            {"in_tensor", in_reg, in_strides});
        in_register_tensor_pointers.push_back(
            {"out_tensor", out_reg, out_strides});
    }

    int possibly_inject_a_loop()
    {
        auto padded_sizes = sizes;
        padded_sizes[vectorized_var] =
            round_up(padded_sizes[vectorized_var], vector_size);

        std::map<std::string, std::vector<int>> ranges;
        for (auto const& p : padded_sizes)
        {
            ranges[p.first].push_back(p.second);
        }

        std::int64_t total_required_move_operations =
            std::accumulate(
                padded_sizes.begin(), padded_sizes.end(), (std::int64_t)1,
                [&](std::int64_t v, auto const& s) { return v * s.second; }) /
            vector_size;

        LN_LOG(DEBUG) << "MOVES: " << total_required_move_operations << "\n";

        // auto sizes_copy = padded_sizes;

        auto it_end = --(order.end());
        auto it     = order.begin();

        int first_unrolled_loop = 0;

        it_end = --(order.end());

        for (; total_required_move_operations > max_unrolled_moves &&
               it != it_end;
             ++it)
        {
            if (it->first == vectorized_var)
            {
                total_required_move_operations /=
                    ceil_div(ranges[it->first].back(), vector_size);
                total_required_move_operations *= (it->second / vector_size);
            }
            else
            {
                total_required_move_operations /= ranges[it->first].back();
                total_required_move_operations *= it->second;
            }

            ++first_unrolled_loop;

            LN_LOG(DEBUG) << "   AT LOOP " << first_unrolled_loop
                          << " MOVES: " << total_required_move_operations
                          << "\n";

            ranges[it->first].push_back(it->second);
        }

        if (total_required_move_operations > max_unrolled_moves)
        {
            auto pair = *it;

            pair.second                    = max_unrolled_moves * vector_size;
            total_required_move_operations = max_unrolled_moves;
            ++first_unrolled_loop;

            LN_LOG(DEBUG) << "INJECTING A LOOP (for unroll): " << pair.first
                          << ", " << pair.second << "\n";
            order.insert(it, pair);
            ++nest_depth;
        }

        return first_unrolled_loop;
    }

    // Some information about the nested loops.  It is kept constant.
    // To be extended with more rich info in the future.
    std::vector<loop_descriptor> loops;

    // Limits per nested partition of the variable. This will be used
    // to figure out loop tails when the loop stride doesn't divide
    // the total loop count.  Heavily used in the recursive loop
    // issuing methods.  The back of the vector represents the current
    // limit in the recursion (nest).
    std::map<std::string, std::vector<int>> limits;

    static void print_ld(loop_descriptor const& l)
    {
        LN_LOG(INFO) << "Loop over " << l.var << " from 0 to " << l.end
                     << " by " << l.delta << "\n";
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
                    LN_LOG(INFO)
                        << loops[i].var << " :: " << loops[i].end << "\n";
                    assert((loops[i].end % vector_size) == 0);
                }
            }
        }
    }

    // Another utility member to be used during the recursive loop
    // visiting methods.  Keeps the current coordinate.
    std::map<std::string, int> current_coordinate_cursor;

    int get_cursor_offset(std::map<std::string, int> const& strides)
    {
        int off = 0;
        for (auto const& s : strides)
        {
            off += current_coordinate_cursor[s.first] * s.second;
        }
        return off;
    }

    void collect_unrolled_moves_below_helper(std::vector<move_operation>& ret,
                                             int                          depth)
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
                memory_argument dest{get_cursor_offset(out_strides),
                                     &out_traits, vector_size};
                memory_argument src{get_cursor_offset(in_strides), &in_traits,
                                    vector_size};

                ret.push_back({dest, src});
                current_coordinate_cursor[loop.var] += vector_size;
            }

            if (rest)
            {
                memory_argument dest{get_cursor_offset(out_strides),
                                     &out_traits, rest};
                memory_argument src{get_cursor_offset(in_strides), &in_traits,
                                    rest};

                ret.push_back({dest, src});
            }

            current_coordinate_cursor[loop.var] = saved_coordinate;
        }
        else
        {
            for (int i = 0; i < limits[loop.var].back() / loop.delta; ++i)
            {
                limits[loop.var].push_back(loop.delta);
                collect_unrolled_moves_below_helper(ret, depth + 1);
                limits[loop.var].pop_back();
                current_coordinate_cursor[loop.var] += loop.delta;
            }

            auto tail = limits[loop.var].back() % loop.delta;

            if (tail)
            {
                limits[loop.var].push_back(tail);
                collect_unrolled_moves_below_helper(ret, depth + 1);
                limits[loop.var].pop_back();
            }

            current_coordinate_cursor[loop.var] = saved_coordinate;
        }
    }

    std::vector<move_operation> collect_unrolled_moves_below(int depth)
    {
        std::vector<move_operation> ret;
        collect_unrolled_moves_below_helper(ret, depth);
        return ret;
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx512> || std::is_same_v<R, avx2_plus>>
    issue_unrolled_moves(std::vector<move_operation> const& moves)
    {
        OpMask mask_reg;

        for (auto const& op : moves)
        {
            switch (op.src.traits->access)
            {
            case VECTOR_PACKED:
                if (op.src.mask != vector_size)
                {
                    vmovups(in_out_vmm_reg | tail_mask_reg,
                            ptr[in_reg + op.src.offset * 4]);
                }
                else
                {
                    vmovups(in_out_vmm_reg, ptr[in_reg + op.src.offset * 4]);
                }

                LN_LOG(INFO) << tabs.back() << "MOVE in[" << op.src.offset
                             << " | " << op.src.mask << "] TO REG\n";
                break;

            case VECTOR_STRIDED:
                mask_reg = (op.src.mask == vector_size) ? full_mask_reg
                                                        : tail_mask_reg;

                // mov(rcx, 0xffff);
                // kmovw(in_temp_mask_reg, rcx.cvt32());

                kmovw(in_temp_mask_reg, mask_reg);
                vgatherdps(
                    in_out_vmm_reg | in_temp_mask_reg,
                    ptr[in_reg + op.src.offset * 4 + in_access_strides_reg]);

                LN_LOG(INFO) << tabs.back() << "GATHER in[" << op.src.offset
                             << ", " << in_strides.at(vectorized_var) << " | "
                             << op.src.mask << "] TO REG\n";
                break;

            default:
                assert(false);
            }

            mask_reg =
                (op.dest.mask == vector_size) ? full_mask_reg : tail_mask_reg;

            switch (op.dest.traits->access)
            {
            case VECTOR_PACKED:
                vmovups(ptr[out_reg + op.dest.offset * 4] | mask_reg,
                        in_out_vmm_reg);

                LN_LOG(INFO)
                    << tabs.back() << "MOVE REG TO out[" << op.dest.offset
                    << " | " << op.dest.mask << "]\n";
                break;

            case VECTOR_STRIDED:
                kmovw(out_temp_mask_reg, mask_reg);
                vscatterdps(
                    ptr[out_reg + op.dest.offset * 4 + out_access_strides_reg] |
                        out_temp_mask_reg,
                    in_out_vmm_reg);

                LN_LOG(INFO)
                    << tabs.back() << "SCATTER REG TO out[" << op.dest.offset
                    << ", " << out_strides.at(vectorized_var) << " | "
                    << op.dest.mask << "]\n";
                break;

            default:
                assert(false);
            }
        }
    }

    template <class R = ISA>
    std::enable_if_t<std::is_same_v<R, avx2>>
    issue_unrolled_moves(std::vector<move_operation> const& moves)
    {
        Ymm mask_reg;

        for (auto const& op : moves)
        {
            switch (op.src.traits->access)
            {
            case VECTOR_PACKED:
                if (op.src.mask != vector_size)
                {
                    vmaskmovps(in_out_vmm_reg, tail_mask_reg,
                               ptr[in_reg + op.src.offset * 4]);
                }
                else
                {
                    vmovups(in_out_vmm_reg, ptr[in_reg + op.src.offset * 4]);
                }
                LN_LOG(INFO) << tabs.back() << "MOVE in[" << op.src.offset
                             << " | " << op.src.mask << "] TO REG\n";
                break;

            case VECTOR_STRIDED:
                mask_reg = (op.src.mask == vector_size) ? full_mask_reg
                                                        : tail_mask_reg;

                vmovaps(in_temp_mask_reg, mask_reg);
                vgatherdps(
                    in_out_vmm_reg,
                    ptr[in_reg + op.src.offset * 4 + in_access_strides_reg],
                    in_temp_mask_reg);

                LN_LOG(INFO) << tabs.back() << "GATHER in[" << op.src.offset
                             << ", " << in_strides.at(vectorized_var) << " | "
                             << op.src.mask << "] TO REG\n";

                break;

            default:
                assert(false);
            }

            switch (op.dest.traits->access)
            {
            case VECTOR_PACKED:
                if (op.dest.mask == vector_size)
                {
                    vmovups(ptr[out_reg + op.dest.offset * 4], in_out_vmm_reg);
                }
                else
                {
                    vmaskmovps(ptr[out_reg + op.dest.offset * 4], tail_mask_reg,
                               in_out_vmm_reg);
                }
                LN_LOG(INFO)
                    << tabs.back() << "MOVE REG TO out[" << op.dest.offset
                    << " | " << op.dest.mask << "]\n";
                break;

            case VECTOR_STRIDED:
                scatter_avx2_register(in_out_vmm_reg, op.dest.mask,
                                      out_reg + op.dest.offset * 4,
                                      out_strides.at(vectorized_var) * 4);

                LN_LOG(INFO)
                    << tabs.back() << "SCATTER REG TO out[" << op.dest.offset
                    << ", " << out_strides.at(vectorized_var) << " | "
                    << op.dest.mask << "]\n";
                break;

            default:
                assert(false);
            }
        }
    }

    void issue_loop_helper(int depth, bool save_loop, bool save_ptrs,
                           int unroll_stage)
    {
        LN_LOG(INFO) << tabs.back() << "// DEPTH: " << depth << "\n";

        if (depth == unroll_stage)
        {
            issue_unrolled_moves(collect_unrolled_moves_below(depth));
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
                push(loop_reg);
            }

            if (full_iterations > 0)
            {
                LN_LOG(INFO)
                    << tabs.back() << "FOR: " << var_name << " FROM 0 TO "
                    << loop_end << " BY " << loop.delta << " {\n";
            }

            if (full_iterations > 1)
            {
                mov(loop_reg.cvt32(), full_iterations);
                Label loopLabel;
                L(loopLabel);

                // --------------------------------------------------
                // RECURSION

                limits[loop.var].push_back(loop.delta);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_helper(depth + 1, true, true, unroll_stage);
                tabs.pop_back();
                limits[loop.var].pop_back();
                // --------------------------------------------------
                // RECURSION

                advance_pointers(loop.var, loop.delta);

                dec(loop_reg.cvt32());
                jnz(loopLabel);
            }
            else if (full_iterations == 1)
            {
                // --------------------------------------------------
                // RECURSION
                limits[loop.var].push_back(loop.delta);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_helper(depth + 1, save_loop, (tail > 0) || save_ptrs,
                                  unroll_stage);
                tabs.pop_back();
                limits[loop.var].pop_back();
                // --------------------------------------------------
                // RECURSION

                if (tail)
                {
                    advance_pointers(loop.var, loop.delta);
                }
            }

            if (tail)
            {
                LN_LOG(INFO) << tabs.back() << "TAIL: " << var_name << " OF "
                             << tail << " {\n";

                limits[loop.var].push_back(tail);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_helper(depth + 1, save_loop, save_ptrs,
                                  unroll_stage);
                tabs.pop_back();
                limits[loop.var].pop_back();

                LN_LOG(INFO) << tabs.back() << "} END TAIL\n";
            }

            if (full_iterations > 0)
            {
                LN_LOG(INFO) << tabs.back() << "} END FOR\n";
            }

            if (full_iterations > 1 && save_loop)
            {
                pop(loop_reg);
            }

            if (multiple_iterations && save_ptrs)
            {
                pop_pointers(loop.var);
            }
        }
    }

    void issue_loops(int unroll_stage)
    {
        issue_loop_helper(0, false, false, unroll_stage);
    }

    void initialize_auxiliary_registers()
    {
        auto tail_mask = sizes[vectorized_var] % vector_size;

        if constexpr (std::is_same_v<ISA, avx2>)
        {
            vmovups(full_mask_reg, ptr[rip + mask_label]);
            if (tail_mask)
            {
                vmovups(tail_mask_reg,
                        ptr[rip + mask_label + 4 * (8 - tail_mask)]);
            }
        }
        else
        {
            mov(rcx, (1 << vector_size) - 1);
            kmovw(full_mask_reg, rcx.cvt32());

            if (tail_mask)
            {
                mov(rcx, (1 << tail_mask) - 1);
                kmovw(tail_mask_reg, rcx.cvt32());
            }
        }

        if (in_traits.access == VECTOR_STRIDED)
        {
            vmovups(in_access_strides_reg, ptr[rip + in_access_strides_label]);
        }

        if (out_traits.access == VECTOR_STRIDED)
        {
            vmovups(out_access_strides_reg,
                    ptr[rip + out_access_strides_label]);
        }
    }

public:
    transposer_code_generator(
        std::vector<std::pair<std::string, int>> const& Order,
        std::map<std::string, int> const&               Sizes,
        std::map<std::string, int> const&               Out_strides,
        std::map<std::string, int> const&               In_strides,
        std::optional<int> user_unroll_limit = std::nullopt)
        : order(Order)
        , sizes(Sizes)
        , out_strides(Out_strides)
        , in_strides(In_strides)
        , nest_depth(Order.size())
        , max_unrolled_moves(user_unroll_limit ? *user_unroll_limit
                                               : default_max_unrolled_moves)

    {
        assert(order.size());
        vectorized_var = order.back().first;

        check_representation();

        set_tensor_traits();

        set_in_register_tensor_pointers();

        auto first_unrolled_loop = possibly_inject_a_loop();

        LN_LOG(INFO) << "First unrolled loop: " << first_unrolled_loop << "\n";

        initialize_loops_data();

        assert(first_unrolled_loop < loops.size());

        initialize_auxiliary_registers();

        issue_loops(first_unrolled_loop);

        vzeroupper();
        ret();

        issue_embedded_constants();
    }
};

#    ifndef DABUN_HEADER_ONLY

extern template class transposer_code_generator<avx2>;
extern template class transposer_code_generator<avx2_plus>;
extern template class transposer_code_generator<avx512>;

#    endif

} // namespace x86
} // namespace dabun

#endif
