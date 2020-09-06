#pragma once

#if !defined(ARM_LOOP_NEST)

#include "transposer.h"

#else

#include "code_generator.h"
#include "common.h"
#include "isa.h"
#include "log.h"
#include "math.h"
#include "utils.h"
#include "xbyak.h"

#include <any>
#include <cassert>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace facebook
{
namespace sysml
{
namespace aot
{

template <class ISA>
class transposer_jitter
    : public code_generator<void(float* Out, float const* In)>
{
private:
    static constexpr int vector_size = isa_traits<aarch64>::vector_size;

    using base            = code_generator<void(float*, float const*)>;
    using memory_argument = memory_argument_type<vector_size>;

    struct move_operation
    {
        memory_argument dest, src;
    };

    template <class T>
    void sub_imm(XReg const& srcdst, T imm)
    {
        if (imm == 0)
            return;

        base::sub_imm(srcdst, srcdst, imm, xtmp1);
    }

    template <class T>
    void add_imm(XReg const& srcdst, T imm)
    {
        if (imm == 0)
            return;

        base::add_imm(srcdst, srcdst, imm, xtmp1);
    }

    template <class T>
    void sadd_imm(XReg const& srcdst, T imm)
    {
        if (imm == 0)
            return;

        if (imm > 0)
        {
            add_imm(srcdst, imm);
        }
        else
        {
            sub_imm(srcdst, -imm);
        }
    }

private:
    void meta_push(XReg const& op) { str(op, post_ptr(stack_reg, 8)); }

    void meta_pop(XReg const& op) { ldr(op, pre_ptr(stack_reg, -8)); }

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
        mov(stack_reg, sp);
        // stp(q8, q9, post_ptr(stack_reg, 64));
        // stp(q10, q11, post_ptr(stack_reg, 64));
        // stp(q12, q13, post_ptr(stack_reg, 64));
        // stp(q14, q15, post_ptr(stack_reg, 64));
    }

    void restore_stack()
    {
        // ldp(q14, q15, pre_ptr(stack_reg, 64));
        // ldp(q12, q13, pre_ptr(stack_reg, 64));
        // ldp(q10, q11, pre_ptr(stack_reg, 64));
        // ldp(q8, q9, pre_ptr(stack_reg, 64));
        add(sp, sp, 1024);
        add(sp, sp, 1024);
    }

private:
    static constexpr int default_max_unrolled_moves = 32;

    Reg64 out_reg   = x0;
    Reg64 in_reg    = x1;
    Reg64 loop_reg  = x2;
    Reg64 stack_reg = x3;
    Reg64 xtmp1     = x4;

    Reg64 movReg1 = x5;
    Reg64 movReg2 = x6;

    std::vector<std::pair<std::string, int>> order;
    std::map<std::string, int>               sizes;
    std::map<std::string, int>               out_strides;
    std::map<std::string, int>               in_strides;

    std::string vectorized_var;

    int nest_depth;

    int max_unrolled_moves;

    tensor_traits out_traits;
    tensor_traits in_traits;

    bool is_vectorized;

    // This is temporary until I find a better way for nice logging
    // that allows for easy debugging
    std::vector<std::string> tabs = {""};

    std::vector<in_register_tensor_pointer_type> in_register_tensor_pointers;

private:
    void check_representation()
    {
        // Make sure strides (and sizes) exist for each order variable
        for (auto const& o : order)
        {
            strong_assert(out_strides.count(o.first) > 0);
            strong_assert(in_strides.count(o.first) > 0);
            strong_assert(sizes.count(o.first) > 0);
        }

        std::map<std::string, int> last_step;

        for (auto const& o : order)
        {
            if (last_step.count(o.first))
            {
                strong_assert(last_step[o.first] >= o.second &&
                              "The steps in 'order' need to be non-increasing");
            }
            last_step[o.first] = o.second;
        }

        for (auto const& o : order)
        {
            strong_assert(last_step[o.first] == 1 &&
                          "Last step in order not equal to 1");
        }

        for (auto const& o : order)
        {
            if (o.first == vectorized_var)
            {
                strong_assert(o.second == 1 || (o.second % vector_size == 0));
            }
        }
    }

    bool set_tensor_traits()
    {
        // Strides along the LSD dimension of the compute order
        int in_access_stride  = in_strides.at(order.back().first);
        int out_access_stride = out_strides.at(order.back().first);

        LN_LOG(DEBUG) << "in access stride: " << in_access_stride << "\n";
        LN_LOG(DEBUG) << "out access stride: " << out_access_stride << "\n";

        is_vectorized = in_access_stride == 1 && out_access_stride == 1;

        access_kind in_access_kind  = is_vectorized ? VECTOR_PACKED : SCALAR;
        access_kind out_access_kind = is_vectorized ? VECTOR_PACKED : SCALAR;

        // TODO remove redundant information.
        in_traits = {
            "in",    in_access_kind,   in_reg,
            nullptr, in_access_stride, is_vectorized ? vector_size : 1};

        out_traits = {
            "out",   out_access_kind,   out_reg,
            nullptr, out_access_stride, is_vectorized ? vector_size : 1};

        return is_vectorized;
    }

    void set_in_register_tensor_pointers()
    {
        in_register_tensor_pointers.push_back(
            {"in_tensor", in_reg, in_strides});
        in_register_tensor_pointers.push_back(
            {"out_tensor", out_reg, out_strides});
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

            int vek_size = is_vectorized ? vector_size : 1;

            auto fullIterations = limits[loop.var].back() / vek_size;
            auto rest           = limits[loop.var].back() % vek_size;

            for (int i = 0; i < fullIterations; ++i)
            {
                memory_argument dest{get_cursor_offset(out_strides),
                                     &out_traits, vek_size};
                memory_argument src{get_cursor_offset(in_strides), &in_traits,
                                    vek_size};

                ret.push_back({dest, src});
                current_coordinate_cursor[loop.var] += vek_size;
            }

            if (rest)
            {
                if (rest > vector_size / 2)
                {
                    memory_argument dest{get_cursor_offset(out_strides),
                                         &out_traits, vector_size / 2};
                    memory_argument src{get_cursor_offset(in_strides),
                                        &in_traits, vector_size / 2};

                    ret.push_back({dest, src});
                    current_coordinate_cursor[loop.var] += vector_size / 2;

                    rest -= vector_size / 2;
                }

                if (rest)
                {
                    memory_argument dest{get_cursor_offset(out_strides),
                                         &out_traits, rest};
                    memory_argument src{get_cursor_offset(in_strides),
                                        &in_traits, rest};
                    ret.push_back({dest, src});
                }
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

    // Collects all (unrolled) FMAs below a certain loop in the nest.
    // Assumes that the limits are correctly set for the current loop
    // in the execution tree of the loop nest.  This is to correctly
    // handle the tail cases.
    std::vector<move_operation> collect_unrolled_moves_below(int depth)
    {
        std::vector<move_operation> ret;
        collect_unrolled_moves_below_helper(ret, depth);
        return ret;
    }

    void offsets_to_post_increment(std::vector<move_operation>& moves)
    {
        std::map<int, int> reg_to_location;

        auto process = [&](int reg_id, auto& loc) {
            if (reg_to_location.count(reg_id))
            {
                auto delta = reg_to_location[reg_id] - loc.offset;

                reg_to_location[reg_id] = loc.offset;
                loc.offset              = delta * 4;
            }
            else
            {
                reg_to_location[reg_id] = loc.offset;
                loc.offset              = 0;
            }
        };

        for (auto it = moves.rbegin(); it != moves.rend(); ++it)
        {
            process(in_reg.getIdx(), it->src);
            process(out_reg.getIdx(), it->dest);
        }
    }

    void issue_unrolled_moves(std::vector<move_operation> moves)
    {
        for (auto const& m : moves)
        {
            LN_LOG(INFO) << tabs.back() << "OUT[" << m.dest.offset << "] <- in["
                         << m.src.offset << "]\n";
        }

        LN_LOG(INFO) << tabs.back() << "ISSUING " << moves.size()
                     << " UNROLLED MOVES\n";

        if (moves.size())
        {
            strong_assert(moves[0].src.offset == 0);
            strong_assert(moves[0].dest.offset == 0);
        }

        offsets_to_post_increment(moves);

        int src_loc  = 0;
        int dest_loc = 0;

        auto issue_read = [&](auto const& loc) {
            auto delta = loc.offset;

            LN_LOG(INFO) << tabs.back() << "READ in[" << (src_loc / 4)
                         << "] (delta: " << delta << ")\n";

            src_loc += delta;

            if (is_vectorized)
            {
                switch (loc.mask)
                {
                case 1:
                    if (delta && delta <= 254 && delta >= -256)
                    {
                        ldr(WReg(movReg1.getIdx()), post_ptr(in_reg, delta));
                    }
                    else
                    {
                        ldr(WReg(movReg1.getIdx()), ptr(in_reg));
                        sadd_imm(in_reg, delta);
                    }
                    break;
                case 2:
                    if (delta && delta <= 252 && delta >= -256)
                    {
                        ldr(movReg1, post_ptr(in_reg, delta));
                    }
                    else
                    {
                        ldr(movReg1, ptr(in_reg));
                        sadd_imm(in_reg, delta);
                    }
                    break;
                case 4:
                    if (delta && delta <= 504 && delta >= -512)
                    {
                        ldp(movReg1, movReg2, post_ptr(in_reg, delta));
                    }
                    else
                    {
                        ldp(movReg1, movReg2, ptr(in_reg));
                        sadd_imm(in_reg, delta);
                    }
                    break;
                default:
                    strong_assert(false && "Mask not supported");
                }
            }
            else
            {
                if (delta && delta <= 254 && delta >= -256)
                {
                    ldr(WReg(movReg1.getIdx()), post_ptr(in_reg, delta));
                }
                else
                {
                    ldr(WReg(movReg1.getIdx()), ptr(in_reg));
                    sadd_imm(in_reg, delta);
                }
            }
        };

        auto issue_write = [&](auto const& loc) {
            auto delta = loc.offset;

            LN_LOG(INFO) << tabs.back() << "WRITE out[" << (dest_loc / 4)
                         << "]\n";

            dest_loc += delta;

            if (is_vectorized)
            {
                switch (loc.mask)
                {
                case 1:
                    if (delta && delta <= 254 && delta >= -256)
                    {
                        str(WReg(movReg1.getIdx()), post_ptr(out_reg, delta));
                    }
                    else
                    {
                        str(WReg(movReg1.getIdx()), ptr(out_reg));
                        sadd_imm(out_reg, delta);
                    }
                    break;
                case 2:
                    if (delta && delta <= 252 && delta >= -256)
                    {
                        str(movReg1, post_ptr(out_reg, delta));
                    }
                    else
                    {
                        str(movReg1, ptr(out_reg));
                        sadd_imm(out_reg, delta);
                    }
                    break;
                case 4:
                    if (delta && delta <= 504 && delta >= -512)
                    {
                        stp(movReg1, movReg2, post_ptr(out_reg, delta));
                    }
                    else
                    {
                        stp(movReg1, movReg2, ptr(out_reg));
                        sadd_imm(out_reg, delta);
                    }
                    break;
                default:
                    strong_assert(false && "Mask not supported");
                }
            }
            else
            {
                if (delta && delta <= 254 && delta >= -256)
                {
                    str(WReg(movReg1.getIdx()), post_ptr(out_reg, delta));
                }
                else
                {
                    str(WReg(movReg1.getIdx()), ptr(out_reg));
                    sadd_imm(out_reg, delta);
                }
            }
        };

        for (int i = 0; i < moves.size(); ++i)
        {
            issue_read(moves[i].src);
            issue_write(moves[i].dest);
        }

        sadd_imm(in_reg, -src_loc);
        sadd_imm(out_reg, -dest_loc);
    }

    void issue_loop_helper(int depth, bool save_loop, bool save_ptrs,
                           int unroll_stage)
    {
        LN_LOG(INFO) << tabs.back() << "// DEPTH: " << depth
                     << " US: " << unroll_stage << "\n";

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
                meta_push(loop_reg);
            }

            if (full_iterations > 0)
            {
                LN_LOG(INFO)
                    << tabs.back() << "FOR: " << var_name << " FROM 0 TO "
                    << loop_end << " BY " << loop.delta << " {\n";
            }

            if (full_iterations > 1)
            {
                mov_imm(loop_reg, full_iterations);
                auto loopLabel = make_label();
                L_aarch64(*loopLabel);

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

                sub_imm(loop_reg, 1);
                cbnz(loop_reg, *loopLabel);
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
                meta_pop(loop_reg);
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

public:
    transposer_jitter(std::vector<std::pair<std::string, int>> const& Order,
                      std::map<std::string, int> const&               Sizes,
                      std::map<std::string, int> const& Out_strides,
                      std::map<std::string, int> const& In_strides,
                      std::optional<int> user_unroll_limit = std::nullopt)
        : order(Order)
        , sizes(Sizes)
        , out_strides(Out_strides)
        , in_strides(In_strides)
        , nest_depth(Order.size())
        , max_unrolled_moves(user_unroll_limit ? *user_unroll_limit
                                               : default_max_unrolled_moves)

    {
        strong_assert(order.size());
        vectorized_var = order.back().first;

        check_representation();

        set_tensor_traits();

        set_in_register_tensor_pointers();
        auto first_unrolled_loop = possibly_inject_a_loop();

        if (!is_vectorized)
        {
            vectorized_var = "";
        }

        LN_LOG(INFO) << "First unrolled loop: " << first_unrolled_loop << "\n";

        initialize_loops_data();

        assert(first_unrolled_loop < loops.size());

        // initialize_auxiliary_registers();

        prepare_stack();

        issue_loops(first_unrolled_loop);

        restore_stack();

        ret();
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook

#endif
