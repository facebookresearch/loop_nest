// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// TODO IMPORTANT VECTOR-VECTOR might be wrong because of horizontal
// adds at the end, combined with the MLAs :(

// TODO(important) WHEN LOADING C WITH SCALAR VALUES and compute is
// VECTOR-VECTOR we need to zero out rest of the C_VMM vectors

#pragma once

#include "dabun/arm/loop_nest_fp16.hpp"

#include "dabun/isa.hpp"
#ifdef DABUN_ARCH_AARCH64

#    include "dabun/arm/arithmetic_operation.hpp"
#    include "dabun/arm/configuration.hpp"
#    include "dabun/arm/elementwise_operation.hpp"
#    include "dabun/arm/meta_mnemonics.hpp"
#    include "dabun/arm/multi_vreg.hpp"
#    include "dabun/code_generator/code_generator.hpp"
#    include "dabun/common.hpp"
#    include "dabun/core.hpp"
#    include "dabun/isa.hpp"
#    include "dabun/loop_nest_descriptor.hpp"
#    include "dabun/math.hpp"
#    include "dabun/utility/log.hpp"

#    include <boost/multi_index/composite_key.hpp>
#    include <boost/multi_index/member.hpp>
#    include <boost/multi_index/ordered_index.hpp>
#    include <boost/multi_index_container.hpp>

#    include <any>
#    include <cstdint>
#    include <limits>
#    include <map>
#    include <numeric>
#    include <optional>
#    include <set>
#    include <tuple>
#    include <type_traits>
#    include <variant>
#    include <vector>

namespace dabun
{
namespace arm
{

template <class, bool = false /* is neon_fp16 */>
class loop_nest_code_generator;

template <>
class loop_nest_code_generator<aarch64, false>
    : public code_generator<void(float* C, float const* A, float const* B,
                                 int alpha)>,
      public meta_mnemonics<loop_nest_code_generator<aarch64>>
{
private:
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

private:
    using base =
        code_generator<void(float* C, float const* A, float const* B, int)>;
    using meta_base   = meta_mnemonics<loop_nest_code_generator<aarch64>>;
    using Vmm         = VReg;
    using multi_vregs = multi_vreg<Vmm, SReg, HReg>;

    static constexpr int bytes_per_float = 4;
    static constexpr int vector_size     = isa_traits<aarch64>::vector_size;

    void prepare_stack()
    {
        sub(sp, sp, 1024);
        sub(sp, sp, 1024);
        mov(stack_reg, sp);
        stp(q8, q9, post_ptr(stack_reg, 64));
        stp(q10, q11, post_ptr(stack_reg, 64));
        stp(q12, q13, post_ptr(stack_reg, 64));
        stp(q14, q15, post_ptr(stack_reg, 64));
    }

    void restore_stack()
    {
        ldp(q14, q15, pre_ptr(stack_reg, -64));
        ldp(q12, q13, pre_ptr(stack_reg, -64));
        ldp(q10, q11, pre_ptr(stack_reg, -64));
        ldp(q8, q9, pre_ptr(stack_reg, -64));
        add(sp, sp, 1024);
        add(sp, sp, 1024);
    }

    struct instruction_register
    {
        int number = 0;
        int lane   = vector_size;

        friend bool operator<(instruction_register const& lhs,
                              instruction_register const& rhs)
        {
            return std::tie(lhs.number, lhs.lane) <
                   std::tie(rhs.number, rhs.lane);
        }
    };

    struct fmla_instruction
    {
        instruction_register dst;
        instruction_register left_src;
        instruction_register right_src;

        friend bool operator<(fmla_instruction const& lhs,
                              fmla_instruction const& rhs)
        {
            return std::tie(lhs.dst, lhs.left_src, lhs.right_src) <
                   std::tie(rhs.dst, rhs.left_src, rhs.right_src);
        }
    };

    struct load_instruction
    {
        int vreg;
        int num_lanes;

        tensor_location_t tensor_location;

        friend bool operator<(load_instruction const& lhs,
                              load_instruction const& rhs)
        {
            return std::tie(lhs.vreg, lhs.num_lanes, lhs.tensor_location) <
                   std::tie(rhs.vreg, rhs.num_lanes, rhs.tensor_location);
        }
    };

    struct load_xreg_instruction
    {
        int reg;

        tensor_location_t tensor_location;

        friend bool operator<(load_xreg_instruction const& lhs,
                              load_xreg_instruction const& rhs)
        {
            return std::tie(lhs.reg, lhs.tensor_location) <
                   std::tie(rhs.reg, rhs.tensor_location);
        }
    };

    struct load_wreg_instruction
    {
        int reg;

        tensor_location_t tensor_location;

        friend bool operator<(load_wreg_instruction const& lhs,
                              load_wreg_instruction const& rhs)
        {
            return std::tie(lhs.reg, lhs.tensor_location) <
                   std::tie(rhs.reg, rhs.tensor_location);
        }
    };

    struct ins_xreg_instruction
    {
        int reg;
        int vreg;
        int lane;

        friend bool operator<(ins_xreg_instruction const& lhs,
                              ins_xreg_instruction const& rhs)
        {
            return std::tie(lhs.reg, lhs.vreg, lhs.lane) <
                   std::tie(rhs.reg, rhs.vreg, rhs.lane);
        }
    };

    struct ins_wreg_instruction
    {
        int reg;
        int vreg;
        int lane;

        friend bool operator<(ins_wreg_instruction const& lhs,
                              ins_wreg_instruction const& rhs)
        {
            return std::tie(lhs.reg, lhs.vreg, lhs.lane) <
                   std::tie(rhs.reg, rhs.vreg, rhs.lane);
        }
    };

    struct load_pair_instruction
    {
        int vreg1;
        int vreg2;

        int num_lanes;

        tensor_location_t tensor_location;

        friend bool operator<(load_pair_instruction const& lhs,
                              load_pair_instruction const& rhs)
        {
            return std::tie(lhs.vreg1, lhs.vreg2, lhs.num_lanes,
                            lhs.tensor_location) <
                   std::tie(rhs.vreg1, rhs.vreg2, rhs.num_lanes,
                            rhs.tensor_location);
        }
    };

    using instruction_t =
        std::variant<std::monostate, load_instruction, load_pair_instruction,
                     fmla_instruction, load_wreg_instruction,
                     load_xreg_instruction, ins_wreg_instruction,
                     ins_xreg_instruction>;

    std::deque<std::vector<instruction_t>> instruction_IRs;

private:
    Reg64 CReg_            = x0;
    Reg64 AReg_            = x1;
    Reg64 BReg_            = x2;
    Reg64 alpha_reg_       = x3;
    Reg64 ZeroReg_         = x4;
    Reg64 xtmp1            = x5;
    Reg64 loopReg_         = x7;
    Reg64 stack_reg        = x9;
    Reg64 tmpCReg_         = x10;
    Reg64 tmpAReg_         = x11;
    Reg64 tmpBReg_         = x12;
    Reg64 skip_postop_reg_ = x14;

    int insReg_ = 10;

    std::vector<int> possible_loop_registers = {6,  15, 19, 20, 21, 22, 23,
                                                24, 25, 26, 27, 28, 29};

    std::vector<int> loop_registers;

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

    struct operation_operation
    {
        memory_argument            dest, src1, src2;
        std::map<std::string, int> coordinates;
    };

private:
    // Here we put some default unroll limit.
    static constexpr int default_max_operations_unrolled = 320;

private:
    std::vector<std::pair<std::string, int>> order;
    std::map<std::string, int> const&        sizes;

    std::set<std::string> const& C_formula;
    std::set<std::string> const& A_formula;
    std::set<std::string> const& B_formula;

    std::map<std::string, int> C_strides;
    std::map<std::string, int> A_strides;
    std::map<std::string, int> B_strides;

    int nest_depth;

    // Which can be overriten by the caller.
    int max_operations_unrolled;

    // Tensors are vectorized if they are looped over in the innermost
    // loop and if the appropriate strides are 1.
    bool is_C_vectorized;
    bool is_A_vectorized;
    bool is_B_vectorized;

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

    std::shared_ptr<elementwise_operation<aarch64>> postop;

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

    std::map<int, std::int64_t> sadd_freq;
    std::map<int, int>          delta_xreg_map;

    // Register blocking info (total registers, redundant ones - which
    // get horizontally summed)
    std::pair<int, int> register_blocking_info_;

private:
    static void move_loads(std::vector<instruction_t>& instructions,
                           int                         max_moves = 10)
    {
        for (int i = 1; i < instructions.size(); ++i)
        {
            auto& insn = instructions[i];
            strong_assert(std::holds_alternative<load_instruction>(insn) ||
                          std::holds_alternative<fmla_instruction>(insn) ||
                          std::holds_alternative<std::monostate>(insn));

            if (std::holds_alternative<load_instruction>(instructions[i]))
            {
                auto load = std::get<load_instruction>(instructions[i]);
                for (int pos = i, moves = 0; pos > 0 && moves < max_moves;
                     --pos, ++moves)
                {
                    if (std::holds_alternative<fmla_instruction>(
                            instructions[pos - 1]))
                    {
                        auto operation =
                            std::get<fmla_instruction>(instructions[pos - 1]);
                        if (load.vreg != operation.left_src.number &&
                            load.vreg != operation.right_src.number)
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
    }

    // The to_skip_loads can be used to possibly optimize for older ARMs, which
    // need to issue parallel loads to W/X registers for higher load throighput
    std::vector<instruction_t>
    reorder_instructions(std::vector<instruction_t> instructions_in,
                         int                        to_skip_loads = 1000000000)
    {
        if constexpr (compiled_in_debug_mode)
        {
            for (auto& insn : instructions_in)
            {
                if (std::holds_alternative<load_instruction>(insn))
                {
                    auto& load = std::get<load_instruction>(insn);
                    strong_assert(load.vreg > -1 && load.vreg < 32);
                }
            }
        }

        pair_loads(instructions_in);

        if constexpr (compiled_in_debug_mode)
        {
            for (auto& insn : instructions_in)
            {
                if (std::holds_alternative<load_instruction>(insn))
                {
                    auto& load = std::get<load_instruction>(insn);
                    strong_assert(load.vreg > -1 && load.vreg < 32);
                }
            }
        }

        std::vector<instruction_t> instructions;

        for (auto& insn : instructions_in)
        {
            if (std::holds_alternative<load_instruction>(insn))
            {
                if (--to_skip_loads >= 0)
                {
                    instructions.push_back(insn);
                    continue;
                }
            }
            if (std::holds_alternative<load_pair_instruction>(insn))
            {
                to_skip_loads -= 2;
                if (to_skip_loads >= 0)
                {
                    instructions.push_back(insn);
                    continue;
                }
            }
            if (std::holds_alternative<load_instruction>(insn))
            {
                auto& load = std::get<load_instruction>(insn);
                if (load.num_lanes < vector_size)
                {
                    load.num_lanes = 2;
                    instructions.push_back(load);
                    instructions.push_back(load_wreg_instruction{
                        insReg_,
                        {load.tensor_location.idx,
                         load.tensor_location.offset + 8}});
                    instructions.push_back(
                        ins_wreg_instruction{insReg_, load.vreg, 2});
                }
                else if (load.num_lanes == 4)
                {
                    load.num_lanes = 2;
                    instructions.push_back(load);
                    instructions.push_back(load_xreg_instruction{
                        insReg_,
                        {load.tensor_location.idx,
                         load.tensor_location.offset + 8}});
                    instructions.push_back(
                        ins_xreg_instruction{insReg_, load.vreg, 1});
                }
                else
                {
                    instructions.push_back(insn);
                }
            }
            else if (!std::holds_alternative<std::monostate>(insn))
            {
                instructions.push_back(insn);
            }
        }

        if constexpr (compiled_in_debug_mode)
        {
            for (auto& insn : instructions)
            {
                if (std::holds_alternative<load_instruction>(insn))
                {
                    auto& load = std::get<load_instruction>(insn);
                    strong_assert(load.vreg > -1 && load.vreg < 32);
                }
            }
        }

        std::vector<int> lane_last_seen(64, -1);

        std::vector<int>              num_prev_uses(instructions.size());
        std::vector<std::vector<int>> future_deps(instructions.size());

        auto visit_vreg = [&](int vreg, int at)
        {
            if (lane_last_seen[vreg] != -1)
            {
                future_deps[lane_last_seen[vreg]].push_back(at);
                ++num_prev_uses[at];
            }
            lane_last_seen[vreg] = at;
        };

        for (int i = 0; i < instructions.size(); ++i)
        {
            auto& insn = instructions[i];

            std::visit(overloaded{[&](fmla_instruction const& fmla)
                                  {
                                      // for (int lane = 0; lane <
                                      // fmla.left_src.lane;
                                      //      ++lane)
                                      {
                                          visit_vreg(fmla.left_src.number, i);
                                      }

                                      // for (int lane = 0; lane <
                                      // fmla.right_src.lane;
                                      // ++lane)
                                      {
                                          visit_vreg(fmla.right_src.number, i);
                                      }
                                  },
                                  [&](load_instruction const& load)
                                  {
                                      // for (int lane = 0; lane <
                                      // load.num_lanes; ++lane)
                                      {
                                          visit_vreg(load.vreg, i);
                                      }
                                  },
                                  [&](load_pair_instruction const& load)
                                  {
                                      // for (int lane = 0; lane <
                                      // load.num_lanes; ++lane)
                                      {
                                          visit_vreg(load.vreg1, i);
                                          visit_vreg(load.vreg2, i);
                                      }
                                  },
                                  [&](load_xreg_instruction const& load)
                                  { visit_vreg(32 + load.reg, i); },
                                  [&](load_wreg_instruction const& load)
                                  { visit_vreg(32 + load.reg, i); },
                                  [&](ins_xreg_instruction const& ins)
                                  {
                                      visit_vreg(32 + ins.reg, i);
                                      visit_vreg(ins.vreg, i);
                                  },
                                  [&](ins_wreg_instruction const& ins)
                                  {
                                      visit_vreg(32 + ins.reg, i);
                                      visit_vreg(ins.vreg, i);
                                  },
                                  [&](std::monostate const&) {}},
                       insn);
        }

        std::deque<int> loads_queue;
        std::deque<int> ins_queue;
        std::deque<int> fmla_queue;

        int pattern_idx = 0;

        constexpr int load_delay = 16;

        std::map<int, std::vector<std::function<void()>>> to_enqueue;

        auto enqueue = [&](int i)
        {
            auto const& insn = instructions[i];

            LN_LOG(INFO) << "  ENQUE: ";
            print_instruction(instructions[i]);

            if (std::holds_alternative<fmla_instruction>(insn))
            {
                fmla_queue.push_back(i);
            }
            else if (std::holds_alternative<load_instruction>(insn) ||
                     std::holds_alternative<load_pair_instruction>(insn) ||
                     std::holds_alternative<load_xreg_instruction>(insn) ||
                     std::holds_alternative<load_wreg_instruction>(insn))
            {
                loads_queue.push_back(i);
            }
            else if (std::holds_alternative<ins_xreg_instruction>(insn) ||
                     std::holds_alternative<ins_wreg_instruction>(insn))
            {
                ins_queue.push_back(i);
            }
            else
            {
                strong_assert(std::holds_alternative<std::monostate>(insn));
            }
        };

        std::vector<instruction_t> ret;

        auto issue = [&](int i)
        {
            LN_LOG(INFO) << "ISSUING: ";
            print_instruction(instructions[i]);

            ret.push_back(instructions[i]);
            for (auto r : future_deps[i])
            {
                if (--num_prev_uses[r] == 0)
                {
                    auto const& insn = instructions[i];
                    if (std::holds_alternative<load_instruction>(insn) ||
                        std::holds_alternative<load_pair_instruction>(insn) ||
                        std::holds_alternative<load_xreg_instruction>(insn) ||
                        std::holds_alternative<load_wreg_instruction>(insn))
                    {
                        to_enqueue[pattern_idx + load_delay].push_back(
                            [&, which = r]() { enqueue(which); });
                    }
                    else
                    {
                        enqueue(r);
                    }
                }
                strong_assert(num_prev_uses[r] >= 0);
            }
        };

        for (int i = 0; i < instructions.size(); ++i)
        {
            if (num_prev_uses[i] == 0)
            {
                enqueue(i);
            }
        }

        enum pattern_t
        {
            LOAD = 0,
            FMLA,
            INS
        };

        std::vector<pattern_t> const pattern = {LOAD, INS,  FMLA,
                                                LOAD, FMLA, FMLA};

        while (ret.size() != instructions.size())
        {
            for (auto const& f : to_enqueue[pattern_idx])
            {
                f();
            }

            switch (pattern[pattern_idx % pattern.size()])
            {
            case LOAD:
                if (loads_queue.size())
                {
                    issue(loads_queue.front());
                    loads_queue.pop_front();
                }
                break;
            case INS:
                if (ins_queue.size())
                {
                    issue(ins_queue.front());
                    ins_queue.pop_front();
                }
                break;
            case FMLA:
                if (fmla_queue.size())
                {
                    issue(fmla_queue.front());
                    fmla_queue.pop_front();
                }
                break;
            }

            ++pattern_idx;
        }

        strong_assert(ret.size() == instructions.size());

        return ret; // instructions;
    }

    void interleave_loads(std::vector<instruction_t>& instructions)
    {
        std::map<int, int> pairity;
        std::map<int, int> temp_pair = {{BReg_.getIdx(), tmpBReg_.getIdx()},
                                        {AReg_.getIdx(), tmpAReg_.getIdx()}};

        for (auto& insn : instructions)
        {
            std::visit(
                overloaded{[&](load_pair_instruction& load_pair)
                           {
                               auto ridx = load_pair.tensor_location.idx;
                               if ((pairity[ridx]++) % 2)
                               {
                                   load_pair.tensor_location.idx =
                                       temp_pair[ridx];
                               }
                           },
                           [&](load_instruction& load)
                           {
                               auto ridx = load.tensor_location.idx;
                               if ((pairity[ridx]++) % 2)
                               {
                                   load.tensor_location.idx = temp_pair[ridx];
                               }
                           },
                           [&](load_wreg_instruction& load)
                           {
                               auto ridx = load.tensor_location.idx;
                               if ((pairity[ridx]++) % 2)
                               {
                                   load.tensor_location.idx = temp_pair[ridx];
                               }
                           },
                           [&](load_xreg_instruction& load)
                           {
                               auto ridx = load.tensor_location.idx;
                               if ((pairity[ridx]++) % 2)
                               {
                                   load.tensor_location.idx = temp_pair[ridx];
                               }
                           },
                           [](ins_wreg_instruction const&) {},
                           [](ins_xreg_instruction const&) {},
                           [](fmla_instruction&) {}, [](std::monostate) {}},
                insn);
        }
    }

    // We have available instructions that can load two vectors at the
    // same time, given that they are consecutive in memory. On newer
    // ARM processors, this operations are optimized - so we can merge
    // loads of consecutive vectors
    static void
    pair_loads(std::vector<instruction_t>& instructions,
               int max_total_paired = std::numeric_limits<int>::max())
    {
        if (instructions.size() == 0)
        {
            return;
        }

        std::map<int, int> mappings;

        int num_possibly_paired = 0;

        for (int i = instructions.size() - 1; i >= 0; --i)
        {
            std::visit(
                overloaded{
                    [](load_pair_instruction const&)
                    { strong_assert(false && "Load pair not expected"); },
                    [&](load_instruction load)
                    {
                        if (load.num_lanes == 3)
                        {
                            return;
                        }

                        mappings.erase(load.vreg);

                        for (auto& m : mappings)
                        {
                            strong_assert(
                                std::holds_alternative<load_instruction>(
                                    instructions[m.second]));

                            auto const& next_load = std::get<load_instruction>(
                                instructions[m.second]);

                            strong_assert(next_load.vreg == m.first);

                            if (next_load.tensor_location.idx ==
                                load.tensor_location.idx)
                            {
                                if (next_load.num_lanes == load.num_lanes &&
                                    next_load.vreg != load.vreg &&
                                    next_load.tensor_location.offset ==
                                        load.tensor_location.offset +
                                            load.num_lanes * bytes_per_float)
                                {
                                    mappings.erase(m.first);
                                    ++num_possibly_paired;
                                    break;
                                }
                            }
                        }

                        if (std::holds_alternative<load_instruction>(
                                instructions[i]))
                        {
                            strong_assert(mappings.count(load.vreg) == 0);
                            mappings[load.vreg] = i;
                        }
                    },
                    [&](fmla_instruction const& fml)
                    {
                        mappings.erase(fml.left_src.number);
                        mappings.erase(fml.right_src.number);
                        mappings.erase(fml.dst.number);
                    },
                    [](load_wreg_instruction const&) { strong_assert(false); },
                    [](load_xreg_instruction const&) { strong_assert(false); },
                    [](ins_wreg_instruction const&) { strong_assert(false); },
                    [](ins_xreg_instruction const&) { strong_assert(false); },
                    [](std::monostate) {}},
                instructions[i]);
        }

        int to_skip = max_total_paired >= num_possibly_paired
                          ? 0
                          : num_possibly_paired - max_total_paired;

        int cur_pos = 0;

        for (int i = instructions.size() - 1; i >= 0; --i)
        {
            std::visit(
                overloaded{
                    [](load_pair_instruction const&)
                    { strong_assert(false && "Load pair not expected"); },
                    [&](load_instruction load)
                    {
                        if (load.num_lanes == 3)
                        {
                            return;
                        }

                        mappings.erase(load.vreg);

                        for (auto& m : mappings)
                        {
                            strong_assert(
                                std::holds_alternative<load_instruction>(
                                    instructions[m.second]));

                            auto const& next_load = std::get<load_instruction>(
                                instructions[m.second]);

                            strong_assert(next_load.vreg == m.first);

                            if (next_load.tensor_location.idx ==
                                load.tensor_location.idx)
                            {
                                if (next_load.num_lanes == load.num_lanes &&
                                    next_load.vreg != load.vreg &&
                                    next_load.tensor_location.offset ==
                                        load.tensor_location.offset +
                                            load.num_lanes * bytes_per_float)
                                {
                                    if (cur_pos++ > to_skip)
                                    {
                                        instructions[i] = load_pair_instruction{
                                            load.vreg, next_load.vreg,
                                            load.num_lanes,
                                            load.tensor_location};
                                        instructions[m.second] =
                                            std::monostate();
                                    }

                                    mappings.erase(m.first);
                                    break;
                                }
                            }
                        }

                        if (std::holds_alternative<load_instruction>(
                                instructions[i]))
                        {
                            strong_assert(mappings.count(load.vreg) == 0);
                            mappings[load.vreg] = i;
                        }
                    },
                    [&](fmla_instruction const& fml)
                    {
                        mappings.erase(fml.left_src.number);
                        mappings.erase(fml.right_src.number);
                        mappings.erase(fml.dst.number);
                    },
                    [](load_wreg_instruction const&) { strong_assert(false); },
                    [](load_xreg_instruction const&) { strong_assert(false); },
                    [](ins_wreg_instruction const&) { strong_assert(false); },
                    [](ins_xreg_instruction const&) { strong_assert(false); },
                    [](std::monostate) {}},
                instructions[i]);
        }
    }

    void print_instruction(instruction_t const& insn) const
    {
        std::visit(
            overloaded{[&](load_pair_instruction const& i)
                       {
                           int ptr_reg_idx = i.tensor_location.idx;
                           int ptr_offset  = i.tensor_location.offset;

                           LN_LOG(INFO) << tabs.back() << "::LOAD PAIR Vreg("
                                        << i.vreg1 << " and " << i.vreg2 << ")["
                                        << i.num_lanes << "], X_" << ptr_reg_idx
                                        << "[" << ptr_offset << "]\n";
                       },
                       [&](load_instruction const& i)
                       {
                           int ptr_reg_idx = i.tensor_location.idx;
                           int ptr_offset  = i.tensor_location.offset;

                           LN_LOG(INFO)
                               << tabs.back() << "::LOAD Vreg(" << i.vreg
                               << ")[" << i.num_lanes << "], X_" << ptr_reg_idx
                               << "[" << ptr_offset << "]\n";
                       },
                       [&](fmla_instruction const& fml)
                       {
                           LN_LOG(INFO) << tabs.back() << "::FMLA Vreg("
                                        << fml.dst.number << "), Vreg("
                                        << fml.left_src.number << "), Vreg("
                                        << fml.right_src.number << ") ["
                                        << fml.right_src.lane << "]\n";
                       },
                       [&](load_wreg_instruction const& i) {
                           LN_LOG(INFO)
                               << tabs.back() << "::LOADW W(" << i.reg << ")\n";
                       },
                       [&](load_xreg_instruction const& i) {
                           LN_LOG(INFO)
                               << tabs.back() << "::LOADX X(" << i.reg << ")\n";
                       },
                       [&](ins_wreg_instruction const& i)
                       {
                           LN_LOG(INFO) << tabs.back() << "::INSW W(" << i.reg
                                        << ") -> VReg(" << i.vreg << ")["
                                        << i.lane << "]\n";
                       },
                       [&](ins_xreg_instruction const& i)
                       {
                           LN_LOG(INFO) << tabs.back() << "::INSX X(" << i.reg
                                        << ") -> VReg(" << i.vreg << ")["
                                        << i.lane << "]\n";
                       },

                       [](std::monostate) {}},
            insn);
    }

    void
    print_instructions(std::vector<instruction_t> const& instructions) const
    {
        for (auto const& insn : instructions)
        {
            print_instruction(insn);
        }
    }

    void offsets_to_post_increment(std::vector<instruction_t>& instructions,
                                   int                         num_iterations)
    {
        std::map<int, int> reg_to_location;

        auto process = [&](auto& load)
        {
            if (reg_to_location.count(load.tensor_location.idx))
            {
                auto delta = reg_to_location[load.tensor_location.idx] -
                             load.tensor_location.offset;

                reg_to_location[load.tensor_location.idx] =
                    load.tensor_location.offset;
                load.tensor_location.offset = delta;

                sadd_freq[delta] += num_iterations;
            }
            else
            {
                reg_to_location[load.tensor_location.idx] =
                    load.tensor_location.offset;
                load.tensor_location.offset = 0;
            }
        };

        for (auto it = instructions.rbegin(); it != instructions.rend(); ++it)
        {
            auto& insn = *it;
            if (std::holds_alternative<load_instruction>(insn))
            {
                process(std::get<load_instruction>(insn));
            }
            if (std::holds_alternative<load_pair_instruction>(insn))
            {
                process(std::get<load_pair_instruction>(insn));
            }
            if (std::holds_alternative<load_xreg_instruction>(insn))
            {
                process(std::get<load_xreg_instruction>(insn));
            }
            if (std::holds_alternative<load_wreg_instruction>(insn))
            {
                process(std::get<load_wreg_instruction>(insn));
            }
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
                auto vek_size = vectorized_var == "NONE" ? 1 : vector_size;

                auto fullIterations = limits[loop.var].back() / vek_size;
                auto rest           = limits[loop.var].back() % vek_size;

                for (int i = 0; i < fullIterations; ++i)
                {
                    ret.insert(memory_argument{get_cursor_offset(C_strides),
                                               &C_traits, vek_size,
                                               current_coordinate_cursor});
                    current_coordinate_cursor[loop.var] += vek_size;
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

    void collect_unrolled_operations_below_helper(
        std::vector<operation_operation>& ret, int depth)
    {
        auto const& loop             = loops[depth];
        auto        saved_coordinate = current_coordinate_cursor[loop.var];

        if (depth == nest_depth - 1) // last, vectorized loop
        {
            strong_assert(loop.delta == 1);

            auto vek_size = vectorized_var == "NONE" ? 1 : vector_size;

            auto fullIterations = limits[loop.var].back() / vek_size;
            auto rest           = limits[loop.var].back() % vek_size;

            for (int i = 0; i < fullIterations; ++i)
            {
                memory_argument dest{get_cursor_offset(C_strides), &C_traits,
                                     vek_size};
                memory_argument src1{get_cursor_offset(B_strides), &B_traits,
                                     vek_size};
                memory_argument src2{get_cursor_offset(A_strides), &A_traits,
                                     vek_size};

                ret.push_back({dest, src1, src2, current_coordinate_cursor});
                current_coordinate_cursor[loop.var] += vek_size;
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
                collect_unrolled_operations_below_helper(ret, depth + 1);
                limits[loop.var].pop_back();
                current_coordinate_cursor[loop.var] += loop.delta;
            }

            auto tail = limits[loop.var].back() % loop.delta;

            if (tail)
            {
                limits[loop.var].push_back(tail);
                collect_unrolled_operations_below_helper(ret, depth + 1);
                limits[loop.var].pop_back();
            }

            current_coordinate_cursor[loop.var] = saved_coordinate;
        }
    }

    // Collects all (unrolled) operations below a certain loop in the nest.
    // Assumes that the limits are correctly set for the current loop
    // in the execution tree of the loop nest.  This is to correctly
    // handle the tail cases.
    std::vector<operation_operation>
    collect_unrolled_operations_below(int depth)
    {
        std::vector<operation_operation> ret;
        collect_unrolled_operations_below_helper(ret, depth);
        return ret;
    }

    void collect_default_unrolled_operations_at_helper(
        std::vector<operation_operation>& ret, int cur_depth, int req_depth)
    {
        if (cur_depth == req_depth)
        {
            collect_unrolled_operations_below_helper(ret, cur_depth);
        }
        else
        {
            auto const& loop = loops[cur_depth];
            limits[loop.var].push_back(loop.delta);
            collect_default_unrolled_operations_at_helper(ret, cur_depth + 1,
                                                          req_depth);
            limits[loop.var].pop_back();
        }
    }

    // Collect all (unrolled) operations below the first instance of the
    // loop of given depth in the execution tree.  Each other instance
    // of the loop in the tree at the same depth will contain a subset
    // of the collected operations as it will be in a tail of at least one
    // loop.
    std::vector<operation_operation>
    collect_default_unrolled_operations_at(int depth)
    {
        std::vector<operation_operation> ret;
        collect_default_unrolled_operations_at_helper(ret, 0, depth);
        return ret;
    }

    // Pushes the "followed" pointers (C, A or B, and any extra ons
    // that will be used by the future arbitrary innermost operations)
    // that have strides along the dimension dim.
    void push_pointers(std::string const& dim)
    {
        std::vector<Reg64> to_push;

        for (auto const& ptr : in_register_tensor_pointers)
        {
            if (ptr.strides.count(dim) && ptr.strides.at(dim) != 0)
            {
                LN_LOG(INFO) << tabs.back() << "PUSH " << ptr.name << "(X"
                             << ptr.reg.getIdx() << ")\n";
                to_push.push_back(ptr.reg);
            }
        }

        meta_push(to_push);
    }

    // Similarly pops the pointers
    void pop_pointers(std::string const& dim)
    {
        std::vector<Reg64> to_pop;

        for (auto const& ptr : in_register_tensor_pointers)
        {
            if (ptr.strides.count(dim) && ptr.strides.at(dim) != 0)
            {
                to_pop.push_back(ptr.reg);
            }
        }

        meta_pop(to_pop);

        for (auto it = to_pop.rbegin(); it != to_pop.rend(); ++it)
        {
            LN_LOG(INFO) << tabs.back() << "POP "
                         << "(X" << it->getIdx() << ")\n";
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
                meta_add_imm(ptr.reg,
                             ptr.strides.at(dim) * delta * bytes_per_float);
            }
        }
    };

    void load_scalar(VReg const& vreg, Reg64 const& base, int offset,
                     int increment = 0)
    {
        if (offset)
        {
            meta_add_imm(base, offset);
        }

        if (vectorized_var != "NONE")
        {
            mov(vreg.b16, ZeroVector_.b16);
        }

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
            meta_sub_imm(base, offset);
        }
        if (increment)
        {
            meta_add_imm(base, increment);
        }
    }

    void store_scalar(VReg const& vreg, Reg64 const& base, int offset,
                      int increment = 0)
    {
        if (offset)
        {
            meta_add_imm(base, offset);
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
            meta_sub_imm(base, offset);
        }
        if (increment)
        {
            meta_add_imm(base, increment);
        }
    }

    void load_vector(VReg const& vreg, Reg64 const& base, int offset, int mask,
                     int increment = 0)
    {
        if (offset)
        {
            meta_add_imm(base, offset);
        }

        strong_assert(mask > 0 && mask <= vector_size);

        auto issue_the_load_instruction = [&](auto const& r)
        {
            if (increment && increment < 256)
            {
                ldr(r, post_ptr(base, increment));
                increment = 0;
            }
            else
            {
                ldr(r, ptr(base));
            }
        };

        if (mask > 2)
        {
            issue_the_load_instruction(QReg(vreg.s4.getIdx()));
        }
        else if (mask > 1)
        {
            issue_the_load_instruction(DReg(vreg.s4.getIdx()));
        }
        else if (mask > 0)
        {
            issue_the_load_instruction(SReg(vreg.s4.getIdx()));
        }

        if (offset)
        {
            meta_sub_imm(base, offset);
        }
        if (increment)
        {
            meta_add_imm(base, increment);
        }
    }

    void store_vector(VReg const& vreg, Reg64 const& base, int offset, int mask,
                      int increment = 0)
    {
        if (offset)
        {
            meta_add_imm(base, offset);
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
            // TODO(zi) BETTER
            st1(vreg.s4[0], ptr(base));
            for (int i = 1; i < mask; ++i)
            {
                meta_add_imm(base, 4);
                offset += 4;
                st1(vreg.s4[i], ptr(base));
            }
        }

        if (offset)
        {
            meta_sub_imm(base, offset);
        }
        if (increment)
        {
            meta_add_imm(base, increment);
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
            meta_add_imm(base, offset);
        }

        st1(vreg.s4[0], ptr(base));

        for (int i = 1; i < mask; ++i)
        {
            meta_add_imm(base, stride);
            offset += stride;
            st1(vreg.s4[i], ptr(base));
        }

        if (offset)
        {
            meta_sub_imm(base, offset);
        }
        if (increment)
        {
            meta_add_imm(base, increment);
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
            meta_add_imm(base, offset);
        }

        ld1(vreg.s4[0], ptr(base));

        for (int i = 1; i < mask; ++i)
        {
            meta_add_imm(base, stride);
            offset += stride;
            ld1(vreg.s4[i], ptr(base));
        }

        if (offset)
        {
            meta_sub_imm(base, offset);
        }
        if (increment)
        {
            meta_add_imm(base, increment);
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
                  [](const memory_argument& a, const memory_argument& b)
                  { return a.offset < b.offset; });

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
        strong_assert(ordered_loads.size());
        meta_add_imm(tmpCReg_, ordered_loads.front().offset * bytes_per_float);

        for (auto const& c : ordered_loads)
        {
            LN_LOG(INFO) << tabs.back() << "LOAD " << c.readable() << "\n";

            // Move the reg pointer
            auto incr = 0;
            if (incrs.size())
            {
                incr = incrs.front() * bytes_per_float;
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
                              C_traits.innermost_stride * bytes_per_float,
                              incr);
                break;
            }

            // Set auxiliary horizontal vector regs to zero
            for (int s = 1; s < C_VMMs[c].size(); ++s)
            {
                mov(C_VMMs[c][s].b16, ZeroVector_.b16);
            }
        }
    }

    void issue_C_loads(std::set<memory_argument> const& loads,
                       bool                             issue_first_alpha_logic)
    {
        for (auto& CVmm : C_VMMs)
        {
            CVmm.second.reset();
        }

        if (issue_first_alpha_logic)
        {
            auto loadDataLabel = make_label();
            auto doneInitLabel = make_label();

            cbnz(alpha_reg_, *loadDataLabel);

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
                        std::optional<int> /* tail_mask */, int max_alpha,
                        bool /* issue_max_alpha_logic */)
    {
        std::vector<memory_argument> ordered_stores;

        for (auto const& c : stores)
        {
            ordered_stores.emplace_back(c);
        }

        std::sort(ordered_stores.begin(), ordered_stores.end(),
                  [](const memory_argument& a, const memory_argument& b)
                  { return a.offset < b.offset; });

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
        strong_assert(ordered_stores.size());
        meta_add_imm(tmpCReg_, ordered_stores.front().offset * bytes_per_float);

        for (auto const& c : ordered_stores)
        {
            LN_LOG(INFO) << tabs.back() << "STORE " << c.readable() << " [["
                         << C_VMMs[c][0].getIdx() << "-"
                         << C_VMMs[c][0].getIdx() + C_VMMs[c].size() - 1
                         << "]]\n";

            C_VMMs[c].reduce<float>(*this);
        }
        for (auto const& c : ordered_stores)
        {
            if (C_traits.access == SCALAR && vectorized_var != "NONE")
            {
                C_VMMs[c].full_reduce<float>(*this, 4);
            }
        }

        if (postop && postop->is_relu())
        {
            auto donePostOpLabel = make_label();

            meta_cmp(alpha_reg_, max_alpha - 1);
            b(Xbyak::LT, *donePostOpLabel);
            tbnz(skip_postop_reg_, 1, *donePostOpLabel);

            for (auto const& c : ordered_stores)
            {
                if (C_traits.access == SCALAR)
                {
                    fmax(SReg(C_VMMs[c][0].getIdx()),
                         SReg(C_VMMs[c][0].getIdx()),
                         SReg(ZeroVector_.getIdx()));
                }
                else
                {
                    switch (c.mask)
                    {
                    case 1:
                        fmax(SReg(C_VMMs[c][0].getIdx()),
                             SReg(C_VMMs[c][0].getIdx()),
                             SReg(ZeroVector_.getIdx()));
                        break;
                    case 2:
                        smax(C_VMMs[c][0].s2, C_VMMs[c][0].s2, ZeroVector_.s2);
                        break;
                    case 3: // fall through
                    case 4:
                        smax(C_VMMs[c][0].s4, C_VMMs[c][0].s4, ZeroVector_.s4);
                        break;
                    default:
                        strong_assert(false && "bad c.mask");
                    }
                }
            }

            L_aarch64(*donePostOpLabel);

            for (auto const& c : ordered_stores)
            {
                auto incr = 0;
                if (incrs.size())
                {
                    incr = incrs.front() * bytes_per_float;
                    incrs.erase(incrs.begin());
                }

                switch (C_traits.access)
                {
                case SCALAR:
                    store_scalar(C_VMMs[c][0], tmpCReg_, 0, incr);
                    break;

                case VECTOR_PACKED:
                    store_vector(C_VMMs[c][0], tmpCReg_, 0, c.mask, incr);
                    break;

                case VECTOR_STRIDED:
                    scatter_vector(C_VMMs[c][0], tmpCReg_, 0, c.mask,
                                   C_traits.innermost_stride * bytes_per_float,
                                   incr);
                    break;
                }
            }
        }
        else
        {
            for (auto const& c : ordered_stores)
            {
                LN_LOG(INFO) << tabs.back() << "STORE " << c.readable() << "\n";

                auto incr = 0;
                if (incrs.size())
                {
                    incr = incrs.front() * bytes_per_float;
                    incrs.erase(incrs.begin());
                }

                switch (C_traits.access)
                {
                case SCALAR:
                    store_scalar(C_VMMs[c][0], tmpCReg_, 0, incr);
                    break;

                case VECTOR_PACKED:
                    store_vector(C_VMMs[c][0], tmpCReg_, 0, c.mask, incr);
                    break;

                case VECTOR_STRIDED:
                    scatter_vector(C_VMMs[c][0], tmpCReg_, 0, c.mask,
                                   C_traits.innermost_stride * bytes_per_float,
                                   incr);
                    break;
                }
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

    void issue_unrolled_operations_vector_vector(
        std::vector<operation_operation> /* operations */,
        std::function<void()> const& epilogue_fn)
    {
        auto instructions = std::move(instruction_IRs.front());
        instruction_IRs.pop_front();

        std::map<int, int> tensor_offsets;

        for (auto const& insn : instructions)
        {
            std::visit(
                overloaded{
                    [&](load_pair_instruction const& i)
                    {
                        strong_assert(i.num_lanes != 3);

                        int ptr_reg_idx = i.tensor_location.idx;
                        int delta       = i.tensor_location.offset;

                        tensor_offsets[ptr_reg_idx] += delta;

                        if (C_traits.access == SCALAR && i.num_lanes == 1)
                        {
                            ins(VReg(i.vreg1).s[1], WReg(ZeroReg_.getIdx()));
                            ins(VReg(i.vreg2).s[1], WReg(ZeroReg_.getIdx()));
                        }

                        strong_assert(delta % (bytes_per_float * i.num_lanes) ==
                                      0);

                        switch (i.num_lanes)
                        {
                        case 1:
                            meta_ldp_post_ptr(SReg(i.vreg1), SReg(i.vreg2),
                                              XReg(ptr_reg_idx), delta);
                            break;
                        case 2:
                            meta_ldp_post_ptr(DReg(i.vreg1), DReg(i.vreg2),
                                              XReg(ptr_reg_idx), delta);
                            break;
                        case 3:
                            strong_assert(false &&
                                          "Num of lanes not supported here");
                            break;
                        case 4:
                            meta_ldp_post_ptr(QReg(i.vreg1), QReg(i.vreg2),
                                              XReg(ptr_reg_idx), delta);
                            break;
                        default:
                            strong_assert(false && "Unknown number of lanes");
                        }
                    },
                    [&](load_instruction const& i)
                    {
                        int ptr_reg_idx = i.tensor_location.idx;
                        int delta       = i.tensor_location.offset;

                        tensor_offsets[ptr_reg_idx] += delta;

                        switch (i.num_lanes)
                        {
                        case 1:
                            meta_ldr_post_ptr(SReg(i.vreg), XReg(ptr_reg_idx),
                                              delta);
                            if (C_traits.access == SCALAR)
                            {
                                ins(VReg(i.vreg).s[1], WReg(ZeroReg_.getIdx()));
                            }

                            break;
                        case 2:
                            meta_ldr_post_ptr(DReg(i.vreg), XReg(ptr_reg_idx),
                                              delta);
                            break;
                        case 3:
                            meta_ldr_post_ptr(QReg(i.vreg), XReg(ptr_reg_idx),
                                              delta);
                            if (C_traits.access == SCALAR)
                            {
                                ins(VReg(i.vreg).s[3], WReg(ZeroReg_.getIdx()));
                            }
                            break;
                        case 4:
                            meta_ldr_post_ptr(QReg(i.vreg), XReg(ptr_reg_idx),
                                              delta);
                            break;
                        default:
                            strong_assert(false && "Unknown number of lanes");
                        }
                    },
                    [&](fmla_instruction const& fml)
                    {
                        switch (fml.left_src.lane)
                        {
                        case 1:
                            // fallthrough
                        case 2:
                            fmla(VReg(fml.dst.number).s2,
                                 VReg(fml.left_src.number).s2,
                                 VReg(fml.right_src.number).s2);
                            break;
                        case 3:
                            // fallthrough
                        case 4:
                            fmla(VReg(fml.dst.number).s4,
                                 VReg(fml.left_src.number).s4,
                                 VReg(fml.right_src.number).s4);
                            break;
                        default:
                            strong_assert(false && "Unknown number of lanes");
                        }
                    },
                    [](load_wreg_instruction const&) { strong_assert(false); },
                    [](load_xreg_instruction const&) { strong_assert(false); },
                    [](ins_wreg_instruction const&) { strong_assert(false); },
                    [](ins_xreg_instruction const&) { strong_assert(false); },
                    [](std::monostate) {}},
                insn);
        }

        print_instructions(instructions);

        for (auto const& offs : tensor_offsets)
        {
            meta_sadd_imm(XReg(offs.first), -offs.second);
        }

        epilogue_fn();
    }

    void issue_unrolled_operations_scalar_scalar(
        std::vector<operation_operation> /* operations */,
        std::function<void()> const& epilogue_fn)
    {
        auto instructions = std::move(instruction_IRs.front());
        instruction_IRs.pop_front();

        std::map<int, int> tensor_offsets;

        for (auto const& insn : instructions)
        {
            std::visit(
                overloaded{
                    [&](load_pair_instruction const& i)
                    {
                        int ptr_reg_idx = i.tensor_location.idx;
                        int delta       = i.tensor_location.offset;

                        tensor_offsets[ptr_reg_idx] += delta;
                        strong_assert(i.num_lanes == 1);

                        meta_ldp_post_ptr(SReg(i.vreg1), SReg(i.vreg2),
                                          XReg(ptr_reg_idx), delta);
                    },
                    [&](load_instruction const& i)
                    {
                        int ptr_reg_idx = i.tensor_location.idx;
                        int delta       = i.tensor_location.offset;

                        tensor_offsets[ptr_reg_idx] += delta;
                        strong_assert(i.num_lanes == 1);

                        meta_ldr_post_ptr(SReg(i.vreg), XReg(ptr_reg_idx),
                                          delta);
                    },
                    [&](fmla_instruction const& fml)
                    {
                        fmla(VReg(fml.dst.number).s2,
                             VReg(fml.left_src.number).s2,
                             VReg(fml.right_src.number).s2);
                    },
                    [](load_wreg_instruction const&) { strong_assert(false); },
                    [](load_xreg_instruction const&) { strong_assert(false); },
                    [](ins_wreg_instruction const&) { strong_assert(false); },
                    [](ins_xreg_instruction const&) { strong_assert(false); },

                    [](std::monostate) {}},
                insn);
        }

        print_instructions(instructions);

        for (auto const& offs : tensor_offsets)
        {
            meta_sadd_imm(XReg(offs.first), -offs.second);
        }

        epilogue_fn();
    }

    void issue_unrolled_operations_scalar_vector(
        std::vector<operation_operation> /* operations */,
        std::function<void()> const& epilogue_fn)
    {

        auto instructions = std::move(instruction_IRs.front());
        instruction_IRs.pop_front();

        std::map<int, int> tensor_offsets;

        mov(tmpBReg_, BReg_);
        mov(tmpAReg_, AReg_);

        int till_epilogue = static_cast<int>(instructions.size());

        while (till_epilogue > 0 && (std::holds_alternative<fmla_instruction>(
                                         instructions[till_epilogue - 1]) ||
                                     std::holds_alternative<std::monostate>(
                                         instructions[till_epilogue - 1])))
        {
            --till_epilogue;
        }

        auto full_epilogue = [&]()
        {
            for (auto const& offs : tensor_offsets)
            {
                if (offs.first == BReg_.getIdx() ||
                    offs.first == AReg_.getIdx())
                {
                    meta_sadd_imm(XReg(offs.first), -offs.second);
                }
            }

            epilogue_fn();
        };

        for (auto const& insn : instructions)
        {
            if (till_epilogue-- == 0)
            {
                full_epilogue();
            }

            std::visit(
                overloaded{
                    [&](load_pair_instruction const& i)
                    {
                        strong_assert(i.num_lanes != 3);

                        int ptr_reg_idx = i.tensor_location.idx;
                        int delta       = i.tensor_location.offset;

                        tensor_offsets[ptr_reg_idx] += delta;

                        switch (i.num_lanes)
                        {
                        case 1:
                            meta_ldp_post_ptr(SReg(i.vreg1), SReg(i.vreg2),
                                              XReg(ptr_reg_idx), delta);
                            // TODO(zi) check whether we need to
                            // insert 0 to reg[1] here.
                            break;
                        case 2:
                            meta_ldp_post_ptr(DReg(i.vreg1), DReg(i.vreg2),
                                              XReg(ptr_reg_idx), delta);
                            break;
                        case 3:
                            strong_assert(false &&
                                          "Num of lanes not supported here");
                            break;
                        case 4:
                            meta_ldp_post_ptr(QReg(i.vreg1), QReg(i.vreg2),
                                              XReg(ptr_reg_idx), delta);
                            break;
                        default:
                            strong_assert(false && "Unknown number of lanes");
                        }
                    },
                    [&](load_instruction const& i)
                    {
                        int ptr_reg_idx = i.tensor_location.idx;
                        int delta       = i.tensor_location.offset;

                        tensor_offsets[ptr_reg_idx] += delta;

                        if (i.num_lanes == 1)
                        {
                            meta_ldr_post_ptr(SReg(i.vreg), XReg(ptr_reg_idx),
                                              delta);
                        }
                        else if (i.num_lanes == 2)
                        {
                            meta_ldr_post_ptr(DReg(i.vreg), XReg(ptr_reg_idx),
                                              delta);
                        }
                        else
                        {
                            meta_ldr_post_ptr(QReg(i.vreg), XReg(ptr_reg_idx),
                                              delta);
                        }
                    },
                    [&](fmla_instruction const& fml)
                    {
                        if (fml.right_src.lane != vector_size)
                        {
                            fmla(VReg(fml.dst.number).s4,
                                 VReg(fml.left_src.number).s4,
                                 VReg(fml.right_src.number)
                                     .s[fml.right_src.lane]);
                        }
                        else
                        {
                            fmla(VReg(fml.dst.number).s4,
                                 VReg(fml.left_src.number).s4,
                                 VReg(fml.right_src.number).s4);
                        }
                    },
                    [&](load_wreg_instruction const& i)
                    {
                        int ptr_reg_idx = i.tensor_location.idx;
                        int delta       = i.tensor_location.offset;

                        tensor_offsets[ptr_reg_idx] += delta;

                        meta_ldr_post_ptr(WReg(i.reg), XReg(ptr_reg_idx),
                                          delta);
                    },
                    [&](load_xreg_instruction const& i)
                    {
                        int ptr_reg_idx = i.tensor_location.idx;
                        int delta       = i.tensor_location.offset;

                        tensor_offsets[ptr_reg_idx] += delta;

                        meta_ldr_post_ptr(XReg(i.reg), XReg(ptr_reg_idx),
                                          delta);
                    },
                    [&](ins_wreg_instruction const& i)
                    { base::ins(VReg(i.vreg).s[i.lane], WReg(i.reg)); },
                    [&](ins_xreg_instruction const& i)
                    { base::ins(VReg(i.vreg).d[i.lane], XReg(i.reg)); },

                    [](std::monostate) {}},
                insn);
        }

        if (till_epilogue == 0)
        {
            full_epilogue();
        }

        print_instructions(instructions);
    }

    void issue_unrolled_operations(std::vector<operation_operation> operations,
                                   std::function<void()> const&     epilogue_fn)
    {
        if (operations.size())
        {
            if (operations[0].src1.traits->access == SCALAR &&
                operations[0].src2.traits->access == VECTOR_PACKED)
            {
                issue_unrolled_operations_scalar_vector(std::move(operations),
                                                        epilogue_fn);
                return;
            }
            else if (operations[0].src1.traits->access == VECTOR_PACKED &&
                     operations[0].src2.traits->access == SCALAR)
            {
                for (auto& f : operations)
                {
                    std::swap(f.src1, f.src2);
                }
                issue_unrolled_operations_scalar_vector(std::move(operations),
                                                        epilogue_fn);
                return;
            }
            else if (operations[0].src1.traits->access == VECTOR_PACKED &&
                     operations[0].src2.traits->access == VECTOR_PACKED)
            {
                issue_unrolled_operations_vector_vector(std::move(operations),
                                                        epilogue_fn);
                return;
            }
            else
            {
                issue_unrolled_operations_scalar_scalar(std::move(operations),
                                                        epilogue_fn);
                return;
            }
        }
        else
        {
            strong_assert(false && "Possibly some dimensions are 0");
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

        if (A_access_kind == VECTOR_STRIDED || B_access_kind == VECTOR_STRIDED)
        {
            C_access_kind = A_access_kind = B_access_kind = SCALAR;
            is_C_vectorized = is_A_vectorized = is_B_vectorized = false;
            is_A_vectorized = is_B_vectorized = is_C_vectorized = false;
            C_access_stride = B_access_stride = A_access_stride = 0;
            vectorized_var                                      = "NONE";
        }

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

    // Returns the first loop that can hold C in register file, and
    // the first loop to be unrolled.
    std::tuple<int, int, int> possibly_inject_a_loop()
    {
        auto vek_size = vectorized_var == "NONE" ? 1 : vector_size;

        auto padded_sizes = sizes;
        padded_sizes[order.back().first] =
            round_up(padded_sizes[order.back().first], vek_size);

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

        std::int64_t total_required_innermost_operations =
            std::accumulate(padded_sizes.begin(), padded_sizes.end(),
                            (std::int64_t)1,
                            [&](std::int64_t v, auto const& s)
                            {
                                return (B_strides.count(s.first) ||
                                        A_strides.count(s.first) ||
                                        C_strides.count(s.first))
                                           ? v * s.second
                                           : v;
                            }) /
            vek_size;

        LN_LOG(DEBUG) << "REGS REQUIRED: " << registers_required
                      << " OPERATIONS: " << total_required_innermost_operations
                      << "\n";

        int first_loop_that_can_hold_C = 0;

        LN_LOG(DEBUG) << "Registers originally required: " << registers_required
                      << "\n";
        LN_LOG(DEBUG) << "C_access_len: " << C_traits.access_len << "\n";

        auto it_end = --(order.end());
        auto it     = order.begin();

        for (; registers_required > available_registers && it != it_end; ++it)
        {
            if (C_formula.count(it->first))
            {
                if (is_C_vectorized && it->first == vectorized_var)
                {
                    registers_required /= (ranges[it->first].back() / vek_size);
                    registers_required *= (it->second / vek_size);
                }
                else
                {
                    registers_required /= ranges[it->first].back();
                    registers_required *= it->second;
                }
            }

            if (it->first == vectorized_var)
            {
                total_required_innermost_operations /=
                    ceil_div(ranges[it->first].back(), vek_size);
                total_required_innermost_operations *= (it->second / vek_size);
            }
            else
            {
                total_required_innermost_operations /= ranges[it->first].back();
                total_required_innermost_operations *= it->second;
            }

            ++first_loop_that_can_hold_C;

            LN_LOG(DEBUG) << "    AT LOOP " << first_loop_that_can_hold_C
                          << " REGS REQUIRED: " << registers_required
                          << " OPERATIONS: "
                          << total_required_innermost_operations << "\n";

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
                    total_required_innermost_operations /=
                        (it->second / vek_size);
                    total_required_innermost_operations *=
                        ceil_div(ranges[it->first].back(), vek_size);
                }
                else
                {
                    total_required_innermost_operations /= it->second;
                    total_required_innermost_operations *=
                        ranges[it->first].back();
                }

                --it;
                --first_loop_that_can_hold_C;
            }

            auto pair = *it;

            int register_limit =
                (it == it_end
                     ? std::min(available_registers, max_operations_unrolled)
                     : available_registers);

            // TODO(zi) MAYBE - increase max_operations_unrolled to
            // available_registers?  There's probably never need
            // to request smaller unroll amount than the number of
            // available registers.
            pair.second =
                register_limit *
                ((pair.first == vectorized_var && is_C_vectorized) ? vek_size
                                                                   : 1);

            registers_required = register_limit;

            LN_LOG(DEBUG) << "INJECTING A LOOP: " << pair.first << ", "
                          << pair.second << "\n";
            it = order.insert(it, pair);

            if (it->first == vectorized_var)
            {
                total_required_innermost_operations /=
                    ceil_div(ranges[it->first].back(), vek_size);
                total_required_innermost_operations *= (it->second / vek_size);
            }
            else
            {
                total_required_innermost_operations /= ranges[it->first].back();
                total_required_innermost_operations *= it->second;
            }

            ++first_loop_that_can_hold_C;

            LN_LOG(DEBUG) << "REVISED AT LOOP " << first_loop_that_can_hold_C
                          << " REGS REQUIRED: " << registers_required
                          << " OPERATIONS: "
                          << total_required_innermost_operations << "\n";

            ranges[it->first].push_back(it->second);

            ++nest_depth;
        }

        int first_unrolled_loop = first_loop_that_can_hold_C;

        it_end = --(order.end());

        for (; total_required_innermost_operations > max_operations_unrolled &&
               it != it_end;
             ++it)
        {
            if (it->first == vectorized_var)
            {
                total_required_innermost_operations /=
                    ceil_div(ranges[it->first].back(), vek_size);
                total_required_innermost_operations *= (it->second / vek_size);
            }
            else
            {
                total_required_innermost_operations /= ranges[it->first].back();
                total_required_innermost_operations *= it->second;
            }

            ++first_unrolled_loop;

            LN_LOG(DEBUG) << "   AT LOOP " << first_unrolled_loop
                          << " OPERATIONS: "
                          << total_required_innermost_operations << "\n";

            ranges[it->first].push_back(it->second);
        }

        if (total_required_innermost_operations > max_operations_unrolled)
        {
            auto pair = *it;

            pair.second = max_operations_unrolled * vek_size;
            total_required_innermost_operations = max_operations_unrolled;
            ++first_unrolled_loop;

            LN_LOG(DEBUG) << "INJECTING A LOOP (for unroll): " << pair.first
                          << ", " << pair.second << "\n";
            order.insert(it, pair);
            ++nest_depth;
        }

        return {first_loop_that_can_hold_C, first_unrolled_loop,
                total_required_innermost_operations};
    }

    int assign_vmm_registers(int depth_for_register_blocked_C,
                             int innermost_operations)
    {
        auto collected_load_store =
            collect_default_loads_and_stores_at(depth_for_register_blocked_C);

        // Assign Vector registers to hold C block
        // TODO(zi) better heuristics here
        {
            int next         = auxiliary_registers;
            int per_register = 1;

            int target_regs_for_C = 16;

            if (collected_load_store.size() < target_regs_for_C &&
                innermost_operations > 6 * collected_load_store.size())
            {
                per_register = target_regs_for_C / collected_load_store.size();
            }

            for (auto const& c : collected_load_store)
            {
                LN_LOG(DEBUG) << "LOAD/STORE: " << c.readable() << " ("
                              << per_register << " VMMs)\n";
                C_VMMs[c] = multi_vregs(per_register, next);
                next += per_register;
            }

            strong_assert(next <= isa_traits<aarch64>::total_vector_registers);

            register_blocking_info_ = {next - auxiliary_registers,
                                       per_register};
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
                    strong_assert((loops[i].end % vector_size) == 0);
                }
            }
        }
    }

    std::tuple<int, int> lower_register_blocked_loop(int first_unrolled_loop,
                                                     int innermost_operations)
    {
        int first_loop_that_can_hold_C = first_unrolled_loop;
        while (first_loop_that_can_hold_C > 0 &&
               C_formula.count(loops[first_loop_that_can_hold_C - 1].var) == 0)
        {
            --first_loop_that_can_hold_C;
            // TODO(zi) check math
            auto const& loop      = loops[first_loop_that_can_hold_C];
            int         expansion = loop.end / loop.delta;
            innermost_operations *= expansion;
        }

        LN_LOG(DEBUG) << "LOAD/STORE C MOVED TO LOOP: "
                      << first_loop_that_can_hold_C << " OVER "
                      << loops[first_loop_that_can_hold_C].var << " WITH "
                      << innermost_operations << " INNER operations\n";

        return {first_loop_that_can_hold_C, innermost_operations};
    }

    void issue_loop_helper(
        int depth, bool save_loop, bool save_ptrs,
        int depth_for_register_blocked_C, int unroll_stage,
        bool issue_first_alpha_logic, int max_alpha, bool issue_max_alpha_logic,
        std::function<void()> epilogue_fn = []() {})
    {
        LN_LOG(INFO) << tabs.back() << "// DEPTH: " << depth
                     << " MAX_ALPHA: " << max_alpha << "\n";

        std::vector<operation_operation> unrolled_operations;
        std::set<memory_argument>        collected_load_store;

        if (depth == depth_for_register_blocked_C)
        {
            collected_load_store = collect_loads_and_stores_below(depth);
            issue_C_loads(collected_load_store, issue_first_alpha_logic);
        }

        if (depth == unroll_stage)
        {
            unrolled_operations = collect_unrolled_operations_below(depth);

            if (depth != depth_for_register_blocked_C)
            {
                issue_unrolled_operations(unrolled_operations, epilogue_fn);
                epilogue_fn = []() {};
            }
            else
            {
                issue_unrolled_operations(unrolled_operations, []() {});
            }
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
                if (loop_registers[depth] == -1 && depth > 0)
                {
                    meta_push(loopReg_);
                }
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
                Reg64 loop_reg = loop_registers[depth] == -1
                                     ? loopReg_
                                     : Reg64(loop_registers[depth]);

                meta_mov_imm(loop_reg, full_iterations);
                auto loopLabel = make_label();
                L_aarch64(*loopLabel);

                // --------------------------------------------------
                // RECURSION
                if (depth < depth_for_register_blocked_C &&
                    C_formula.count(loop.var) == 0)
                {
                    new_max_alpha += (full_iterations - 1 + (tail ? 1 : 0)) * 2;
                }

                auto next_epilogue_fn = [&]()
                {
                    advance_pointers(loop.var, loop.delta);

                    if (depth < depth_for_register_blocked_C &&
                        C_formula.count(loop.var) == 0)
                    {
                        meta_add_imm(alpha_reg_, 2);
                    }

                    meta_sub_imm(loop_reg, 1);
                    cmp(loop_reg, 0);
                };

                limits[loop.var].push_back(loop.delta);
                tabs.push_back(tabs.back() + "    ");
                issue_loop_helper(
                    depth + 1, true, true, depth_for_register_blocked_C,
                    unroll_stage, issue_first_alpha_logic, new_max_alpha,
                    recursive_issue_max_alpha_logic /*, next_epilogue_fn*/);
                tabs.pop_back();
                limits[loop.var].pop_back();
                // --------------------------------------------------
                // RECURSION

                next_epilogue_fn();

                // cbnz(loop_reg, *loopLabel);
                b(Xbyak::NE, *loopLabel);
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
                        meta_add_imm(alpha_reg_, 2);
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
                    issue_max_alpha_logic); // TODO(zi) something is
                                            // weird with this logic,
                                            // should work with
                                            // !multiple_iterations &&
                                            // save_ptrs
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
                meta_sub_imm(alpha_reg_, full_iterations * 2);
            }

            if (full_iterations > 1 && save_loop)
            {
                if (loop_registers[depth] == -1 && depth > 0)
                {
                    meta_pop(loopReg_);
                }
            }

            if (multiple_iterations && save_ptrs)
            {
                pop_pointers(loop.var);
            }

            epilogue_fn();
        }

        if (depth == depth_for_register_blocked_C)
        {
            issue_C_stores(collected_load_store, max_alpha,
                           issue_max_alpha_logic);
        }

        epilogue_fn();
    }

    void issue_loops(int depth_for_register_blocked_C, int unroll_stage)
    {
        issue_loop_helper(0, false, false, depth_for_register_blocked_C,
                          unroll_stage, true, 1, true);
    }

    void issue_unrolled_operations_dry_run(
        std::vector<operation_operation> operations, int num_iterations)
    {
        if (operations.size())
        {
            if (operations[0].src1.traits->access == VECTOR_PACKED &&
                operations[0].src2.traits->access == SCALAR)
            {
                for (auto& f : operations)
                {
                    std::swap(f.src1, f.src2);
                }
            }
            else if (operations[0].src1.traits->access == SCALAR &&
                     operations[0].src2.traits->access == VECTOR_PACKED)

            {
                strong_assert(operations[0].src1.traits->access == SCALAR &&
                              operations[0].src2.traits->access ==
                                  VECTOR_PACKED);
            }
            else
            {
                issue_unrolled_operations_dry_run_ss_or_vv(
                    std::move(operations), num_iterations);
                return;
            }
        }

        int src1_reg = operations[0].src1.traits->reg.getIdx();
        int src2_reg = operations[0].src2.traits->reg.getIdx();

        std::map<tensor_location_t, std::deque<int>> remaining_usages;

        for (int i = 0; i < operations.size(); ++i)
        {
            remaining_usages[{src1_reg,
                              operations[i].src1.offset * bytes_per_float}]
                .push_back(i);
            remaining_usages[{src2_reg,
                              operations[i].src2.offset * bytes_per_float}]
                .push_back(i);
        }

        auto num_regs = isa_traits<aarch64>::total_vector_registers -
                        first_unused_vmm_register;

        std::deque<int> free_regs;

        free_regs.push_back(1);
        free_regs.push_back(2);

        for (auto i = 0; i < num_regs; ++i)
        {
            free_regs.push_back(first_unused_vmm_register + i);
        }

        struct table_entry
        {
            tensor_location_t tensor_location;

            int vreg_idx;
            int vreg_lane;
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
                    member<table_entry, int, &table_entry::vreg_idx>
                    >,
                ordered_non_unique<
                    member<table_entry, int, &table_entry::next_usage>,
                    std::greater<int>
                    >
                >
            >;

        // clang-format on

        table_container table;

        auto& tensor_location_index = table.get<0>();
        auto& vreg_index            = table.get<1>();
        auto& next_usage_index      = table.get<2>();

        std::vector<instruction_t> instructions;

        auto load_scalar = [&](int vreg, int tensor_idx, int offset)
        {
            strong_assert(vreg > -1 && vreg < 32);
            load_instruction insn;
            insn.vreg     = vreg;
            int num_lanes = 1;

            tensor_location_t tensor_loc{tensor_idx, offset};

            strong_assert(remaining_usages.count(tensor_loc) &&
                          remaining_usages[tensor_loc].size() > 0);

            int next_usage = remaining_usages[tensor_loc].front();

            table.insert(table_entry{tensor_loc, vreg, 0, next_usage});

            for (int i = 1; i <= 3; ++i)
            {
                tensor_loc.offset += 4;
                if (remaining_usages.count(tensor_loc) &&
                    remaining_usages[tensor_loc].size())
                {
                    int next_usage = remaining_usages[tensor_loc].front();
                    table.insert(table_entry{tensor_loc, vreg, i, next_usage});
                    num_lanes = i + 1;
                }
            }

            // TODO(zi) double check whether we need thbis
            // if (num_lanes == 3)
            // {
            //     num_lanes = 4;
            // }

            insn.num_lanes       = num_lanes;
            insn.tensor_location = {tensor_idx, offset};

            instructions.push_back(insn);
        };

        auto load_vector = [&](int vreg, int tensor_idx, int offset)
        {
            load_instruction insn;
            insn.vreg = vreg;

            tensor_location_t tensor_loc{tensor_idx, offset};

            strong_assert(remaining_usages.count(tensor_loc) &&
                          remaining_usages[tensor_loc].size() > 0);

            int next_usage = remaining_usages[tensor_loc].front();

            table.insert(
                table_entry{tensor_loc, vreg, vector_size, next_usage});

            insn.num_lanes       = vector_size;
            insn.tensor_location = {tensor_idx, offset};

            instructions.push_back(insn);
        };

        auto free_a_register = [&](std::set<int> const& to_avoid)
        {
            auto nu_it = next_usage_index.begin();
            strong_assert(nu_it != next_usage_index.end());

            while (to_avoid.count(nu_it->vreg_idx))
            {
                ++nu_it;
                strong_assert(nu_it != next_usage_index.end());
            }

            int reg_no = nu_it->vreg_idx;

            auto it = vreg_index.find(reg_no);
            while (it != vreg_index.end())
            {
                vreg_index.erase(it);
                it = vreg_index.find(reg_no);
            }

            strong_assert(reg_no > -1 && reg_no < 32);
            return reg_no;
        };

        for (int i = 0; i < operations.size(); ++i)
        {
            tensor_location_t scalar_loc = {
                src1_reg, operations[i].src1.offset * bytes_per_float};
            tensor_location_t vector_loc = {
                src2_reg, operations[i].src2.offset * bytes_per_float};

            int           needs_free_regs = 0;
            std::set<int> to_avoid;

            if (auto it = tensor_location_index.find(scalar_loc);
                it == tensor_location_index.end())
            {
                ++needs_free_regs;
            }
            else
            {
                to_avoid.insert(it->vreg_idx);
            }

            if (auto it = tensor_location_index.find(vector_loc);
                it == tensor_location_index.end())
            {
                ++needs_free_regs;
            }
            else
            {
                to_avoid.insert(it->vreg_idx);
            }

            while (needs_free_regs > free_regs.size())
            {
                free_regs.push_back(free_a_register(to_avoid));
            }

            if (auto it = tensor_location_index.find(scalar_loc);
                it == tensor_location_index.end())
            {
                load_scalar(free_regs.front(), src1_reg,
                            operations[i].src1.offset * bytes_per_float);
                free_regs.pop_front();

                strong_assert(tensor_location_index.find(scalar_loc) !=
                              tensor_location_index.end());
            }

            if (auto it = tensor_location_index.find(vector_loc);
                it == tensor_location_index.end())
            {
                load_vector(free_regs.front(), src2_reg,
                            operations[i].src2.offset * bytes_per_float);
                free_regs.pop_front();

                strong_assert(tensor_location_index.find(vector_loc) !=
                              tensor_location_index.end());
            }

            auto s_it = tensor_location_index.find(scalar_loc);
            auto v_it = tensor_location_index.find(vector_loc);

            strong_assert(s_it != tensor_location_index.end());
            strong_assert(v_it != tensor_location_index.end());

            strong_assert(v_it->vreg_lane >= s_it->vreg_lane);

            // issue OPERATION
            fmla_instruction to_push{
                {(int)((C_VMMs[operations[i].dest]++).getIdx()),
                 v_it->vreg_lane},
                {v_it->vreg_idx, v_it->vreg_lane},
                {s_it->vreg_idx, s_it->vreg_lane}}; // update datastructures

            instructions.push_back(to_push);

            strong_assert(remaining_usages.count(scalar_loc) &&
                          remaining_usages[scalar_loc].size() &&
                          remaining_usages[scalar_loc].front() ==
                              s_it->next_usage);

            strong_assert(remaining_usages.count(vector_loc) &&
                          remaining_usages[vector_loc].size() &&
                          remaining_usages[vector_loc].front() ==
                              v_it->next_usage);

            // Update scalar
            {
                auto s = *s_it;
                tensor_location_index.erase(s_it);
                remaining_usages[scalar_loc].pop_front();
                if (remaining_usages[scalar_loc].size())
                {
                    s.next_usage = remaining_usages[scalar_loc].front();
                    table.insert(s);
                }
                else
                {
                    remaining_usages.erase(scalar_loc);
                    if (vreg_index.find(s.vreg_idx) == vreg_index.end())
                    {
                        free_regs.push_back(s.vreg_idx);
                    }
                }
            }

            // Update vector
            {
                auto v = *v_it;
                tensor_location_index.erase(v_it);
                remaining_usages[vector_loc].pop_front();
                if (remaining_usages[vector_loc].size())
                {
                    v.next_usage = remaining_usages[vector_loc].front();
                    table.insert(v);
                }
                else
                {
                    remaining_usages.erase(vector_loc);
                    if (vreg_index.find(v.vreg_idx) == vreg_index.end())
                    {
                        free_regs.push_back(v.vreg_idx);
                    }
                }
            }
        }

        // move_loads(instructions);
        // pair_loads(instructions);

        instructions = reorder_instructions(std::move(instructions));

        print_instructions(instructions);

        offsets_to_post_increment(instructions, num_iterations);

        instruction_IRs.push_back(std::move(instructions));
    }

    void issue_unrolled_operations_dry_run_ss_or_vv(
        std::vector<operation_operation> operations, int num_iterations)
    {
        int src1_reg = operations[0].src1.traits->reg.getIdx();
        int src2_reg = operations[0].src2.traits->reg.getIdx();

        std::map<tensor_location_t, std::deque<int>> remaining_usages;

        for (int i = 0; i < operations.size(); ++i)
        {
            remaining_usages[{src1_reg,
                              operations[i].src1.offset * bytes_per_float}]
                .push_back(i);
            remaining_usages[{src2_reg,
                              operations[i].src2.offset * bytes_per_float}]
                .push_back(i);
        }

        auto num_regs = isa_traits<aarch64>::total_vector_registers -
                        first_unused_vmm_register;

        std::deque<int> free_regs;

        free_regs.push_back(1);
        free_regs.push_back(2);

        for (auto i = 0; i < num_regs; ++i)
        {
            free_regs.push_back(first_unused_vmm_register + i);
        }

        struct table_entry
        {
            tensor_location_t tensor_location;

            int vreg_idx;
            int vreg_lanes;
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
                    member<table_entry, int, &table_entry::vreg_idx>
                    >,
                ordered_non_unique<
                    member<table_entry, int, &table_entry::next_usage>,
                    std::greater<int>
                    >
                >
            >;

        // clang-format on

        table_container table;

        auto& tensor_location_index = table.get<0>();
        auto& vreg_index            = table.get<1>();
        auto& next_usage_index      = table.get<2>();

        std::vector<instruction_t> instructions;

        auto load_vector =
            [&](int vreg, int tensor_idx, int offset, int num_lanes)
        {
            strong_assert(vreg > -1 && vreg < 32);
            load_instruction insn;
            insn.vreg = vreg;

            tensor_location_t tensor_loc{tensor_idx, offset};

            strong_assert(remaining_usages.count(tensor_loc) &&
                          remaining_usages[tensor_loc].size() > 0);

            int next_usage = remaining_usages[tensor_loc].front();

            table.insert(
                table_entry{tensor_loc, vreg, vector_size, next_usage});

            insn.num_lanes       = num_lanes;
            insn.tensor_location = {tensor_idx, offset};

            instructions.push_back(insn);
        };

        auto free_a_register = [&](std::set<int> const& to_avoid)
        {
            auto nu_it = next_usage_index.begin();
            strong_assert(nu_it != next_usage_index.end());

            while (to_avoid.count(nu_it->vreg_idx))
            {
                ++nu_it;
                strong_assert(nu_it != next_usage_index.end());
            }

            int reg_no = nu_it->vreg_idx;

            auto it = vreg_index.find(reg_no);
            while (it != vreg_index.end())
            {
                vreg_index.erase(it);
                it = vreg_index.find(reg_no);
            }

            strong_assert(reg_no > -1 && reg_no < 32);
            return reg_no;
        };

        auto update_vector = [&](auto v_it)
        {
            auto v = *v_it;
            tensor_location_index.erase(v_it);
            remaining_usages[v.tensor_location].pop_front();
            if (remaining_usages[v.tensor_location].size())
            {
                v.next_usage = remaining_usages[v.tensor_location].front();
                table.insert(v);
            }
            else
            {
                remaining_usages.erase(v.tensor_location);
                if (vreg_index.find(v.vreg_idx) == vreg_index.end())
                {
                    free_regs.push_back(v.vreg_idx);
                }
            }
        };

        for (int i = 0; i < operations.size(); ++i)
        {
            tensor_location_t first_loc = {src1_reg, operations[i].src1.offset *
                                                         vector_size};
            tensor_location_t second_loc = {
                src2_reg, operations[i].src2.offset * bytes_per_float};

            int           needs_free_regs = 0;
            std::set<int> to_avoid;

            if (auto it = tensor_location_index.find(first_loc);
                it == tensor_location_index.end())
            {
                ++needs_free_regs;
            }
            else
            {
                to_avoid.insert(it->vreg_idx);
            }

            if (auto it = tensor_location_index.find(second_loc);
                it == tensor_location_index.end())
            {
                ++needs_free_regs;
            }
            else
            {
                to_avoid.insert(it->vreg_idx);
            }

            while (needs_free_regs > free_regs.size())
            {
                free_regs.push_back(free_a_register(to_avoid));
            }

            if (auto it = tensor_location_index.find(first_loc);
                it == tensor_location_index.end())
            {
                load_vector(free_regs.front(), src1_reg,
                            operations[i].src1.offset * bytes_per_float,
                            operations[i].src1.mask);
                free_regs.pop_front();

                strong_assert(tensor_location_index.find(first_loc) !=
                              tensor_location_index.end());
            }

            if (auto it = tensor_location_index.find(second_loc);
                it == tensor_location_index.end())
            {
                load_vector(free_regs.front(), src2_reg,
                            operations[i].src2.offset * bytes_per_float,
                            operations[i].src2.mask);
                free_regs.pop_front();

                strong_assert(tensor_location_index.find(second_loc) !=
                              tensor_location_index.end());
            }

            auto first_it  = tensor_location_index.find(first_loc);
            auto second_it = tensor_location_index.find(second_loc);

            strong_assert(first_it != tensor_location_index.end());
            strong_assert(second_it != tensor_location_index.end());

            strong_assert(operations[i].src1.mask == operations[i].src2.mask);

            int mask = operations[i].src1.mask;

            // issue OPERATION
            fmla_instruction to_push{
                {(int)((C_VMMs[operations[i].dest]++).getIdx()), mask},
                {first_it->vreg_idx, mask},
                {second_it->vreg_idx, mask}}; // update datastructures

            instructions.push_back(to_push);

            strong_assert(remaining_usages.count(first_loc) &&
                          remaining_usages[first_loc].size() &&
                          remaining_usages[first_loc].front() ==
                              first_it->next_usage);

            strong_assert(remaining_usages.count(second_loc) &&
                          remaining_usages[second_loc].size() &&
                          remaining_usages[second_loc].front() ==
                              second_it->next_usage);

            update_vector(first_it);
            update_vector(second_it);
        }

        move_loads(instructions);
        pair_loads(instructions);
        offsets_to_post_increment(instructions, num_iterations);

        instruction_IRs.push_back(std::move(instructions));
    }

    void issue_loop_dry_run_helper(int depth, int unroll_stage,
                                   int num_iterations)
    {
        LN_LOG(INFO) << tabs.back() << "// DRY_RUN DEPTH: " << depth << "\n";

        std::vector<operation_operation> unrolled_operations;

        if (depth == unroll_stage)
        {
            unrolled_operations = collect_unrolled_operations_below(depth);
            issue_unrolled_operations_dry_run(unrolled_operations,
                                              num_iterations);
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
        // when the innermost loop (vectorized dimension) doesn't fill
        // up a vector register there are lanes that are masked out, we
        // count the FLOPs resulting from these masked out lanes
        std::pair<std::string, int> innermost = order.back();
        // compute the bound for the innermost loop, since this
        // determines the number of elements vectorized identify bound
        // by looking at stride for prior split (if any)
        auto matches_innermost = [&innermost](auto const& dim)
        { return dim.first == innermost.first; };
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
        C_memory_     = C_memory_approx * bytes_per_float;
        A_memory_     = A_memory_approx * bytes_per_float;
        B_memory_     = B_memory_approx * bytes_per_float;
        total_memory_ = C_memory_ + A_memory_ + B_memory_;
    }

    std::vector<Reg64> prepare_loop_registers(int unroll_stage)
    {
        loop_registers     = std::vector<int>(unroll_stage, -1);
        int first_loop_reg = std::max(
            0, unroll_stage - static_cast<int>(possible_loop_registers.size()));

        std::vector<Reg64> to_save;

        for (int i = first_loop_reg; i < loop_registers.size(); ++i)
        {
            loop_registers[i] = possible_loop_registers[i - first_loop_reg];
            if (loop_registers[i] >= 19)
            {
                to_save.push_back(Reg64(loop_registers[i]));
            }
        }

        return to_save;
    }

public:
    std::int64_t get_effective_flops() const { return effective_flops_; }
    std::int64_t get_masked_out_flops() const { return masked_out_flops_; }
    std::int64_t get_total_memory() const { return total_memory_; }

    std::pair<int, int> const& get_register_blocking_info() const
    {
        return register_blocking_info_;
    }

    access_kind get_A_access_kind() const { return A_traits.access; }
    access_kind get_B_access_kind() const { return B_traits.access; }
    access_kind get_C_access_kind() const { return C_traits.access; }

private:
    struct not_depricated_tag
    {
    };

    loop_nest_code_generator(
        not_depricated_tag,
        std::vector<std::pair<std::string, int>> const& _order,
        std::map<std::string, int> const&               sizes,
        std::set<std::string> const&                    C_formula,
        std::set<std::string> const&                    A_formula,
        std::set<std::string> const&                    B_formula,
        std::map<std::string, int> const&               C_strides,
        std::map<std::string, int> const&               A_strides,
        std::map<std::string, int> const&               B_strides,
        std::shared_ptr<operation_pair_base> /*op_pair */,
        std::optional<int> user_operation_unroll_limit = std::nullopt,
        std::shared_ptr<elementwise_operation<aarch64>> /* elementwise_preop
                                                         */
        = nullptr,
        std::vector<std::map<std::string, int>> const&
        /* elementwise_preop_strides */
        = {},
        std::shared_ptr<elementwise_operation<aarch64>> elementwise_postop =
            nullptr,
        std::vector<std::map<std::string, int>> const&
        /* elementwise_postop_strides */
        = {},
        std::optional<OptimizationConfiguration> const& /* optim_config */ =
            std::nullopt)
        : meta_base(x9, x5)
        , order(_order)
        , sizes(sizes)
        , C_formula(C_formula)
        , A_formula(A_formula)
        , B_formula(B_formula)
        , C_strides(C_strides)
        , A_strides(A_strides)
        , B_strides(B_strides)
        , nest_depth(_order.size())
        , max_operations_unrolled(user_operation_unroll_limit
                                      ? *user_operation_unroll_limit
                                      : default_max_operations_unrolled)
        , is_C_vectorized(C_strides.count(order.back().first) == 1)
        , is_A_vectorized(A_strides.count(order.back().first) == 1)
        , is_B_vectorized(B_strides.count(order.back().first) == 1)
        , postop(elementwise_postop)
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

        set_tensor_traits();

        set_available_vector_registers();

        set_in_register_tensor_pointers();

        int first_loop_that_can_hold_C, unroll_stage,
            total_required_innermost_operations;

        std::tie(first_loop_that_can_hold_C, unroll_stage,
                 total_required_innermost_operations) =
            possibly_inject_a_loop();

        initialize_loops_data();

        strong_assert(unroll_stage < loops.size());

        int depth_for_register_blocked_C = first_loop_that_can_hold_C;
        int innermost_operations         = total_required_innermost_operations;

        if (first_loop_that_can_hold_C < unroll_stage)
        {
            std::tie(depth_for_register_blocked_C, innermost_operations) =
                lower_register_blocked_loop(unroll_stage, innermost_operations);
        }

        first_unused_vmm_register = assign_vmm_registers(
            first_loop_that_can_hold_C, innermost_operations);

        //

        std::vector<operation_operation> unrolled_operations =
            collect_default_unrolled_operations_at(unroll_stage);

        strong_assert(unrolled_operations.size() ==
                      total_required_innermost_operations);

        prepare_stack();
        eor(ZeroReg_, ZeroReg_, ZeroReg_);
        ins(ZeroVector_.d[0], ZeroReg_);
        ins(ZeroVector_.d[1], ZeroReg_);

        issue_loops_dry_run(unroll_stage);

        std::vector<int> available = {11, 12, 13, 14, 15};
        std::vector<std::pair<std::int64_t, int>> rev_freq;

        for (auto const& f : sadd_freq)
        {
            if (f.first < -256 || f.first >= 256)
            {
                rev_freq.push_back({f.second, f.first});
            }
            LN_LOG(INFO) << "SADD OF " << f.first << " :: " << f.second << "\n";
        }

        std::sort(rev_freq.begin(), rev_freq.end());

        // while (rev_freq.size() && available.size())
        // {
        //     mov_imm(XReg(available.back()), 0);
        //     sadd_imm(XReg(available.back()), rev_freq.back().second);
        //     delta_xreg_map[rev_freq.back().second] =
        //     available.back(); available.pop_back();
        //     rev_freq.pop_back();
        // }

        auto x_regs_to_save = prepare_loop_registers(unroll_stage);
        meta_push(x_regs_to_save);

        mov(skip_postop_reg_, alpha_reg_);
        and_(alpha_reg_, alpha_reg_, 0x1);

        issue_loops(depth_for_register_blocked_C, unroll_stage);

        meta_pop(x_regs_to_save);

        restore_stack();
        ret();
    }

public:
    // [[deprecated("Use the named argument constructor")]]
    loop_nest_code_generator(
        std::vector<std::pair<std::string, int>> const& _order,
        std::map<std::string, int> const&               sizes,
        std::set<std::string> const&                    C_formula,
        std::set<std::string> const&                    A_formula,
        std::set<std::string> const&                    B_formula,
        std::map<std::string, int> const&               C_strides,
        std::map<std::string, int> const&               A_strides,
        std::map<std::string, int> const&               B_strides,
        std::shared_ptr<operation_pair_base> const&     op_pair,
        std::optional<int> user_operation_unroll_limit = std::nullopt,
        std::shared_ptr<elementwise_operation<aarch64>> elementwise_preop =
            nullptr,
        std::vector<std::map<std::string, int>> const&
            elementwise_preop_strides = {},
        std::shared_ptr<elementwise_operation<aarch64>> elementwise_postop =
            nullptr,
        std::vector<std::map<std::string, int>> const&
            elementwise_postop_strides                        = {},
        std::optional<OptimizationConfiguration> optim_config = std::nullopt)
        : loop_nest_code_generator(
              not_depricated_tag(), _order, sizes, C_formula, A_formula,
              B_formula, C_strides, A_strides, B_strides, op_pair,
              user_operation_unroll_limit, elementwise_preop,
              elementwise_preop_strides, elementwise_postop,
              elementwise_postop_strides, optim_config)
    {
    }

    loop_nest_code_generator(
        loop_nest_verified_descriptor const&        arguments,
        std::shared_ptr<operation_pair_base> const& op_pair,
        std::optional<int> user_operation_unroll_limit = std::nullopt,
        std::shared_ptr<elementwise_operation<aarch64>> elementwise_preop

        = nullptr,
        std::vector<std::map<std::string, int>> const&
            elementwise_preop_strides = {},
        std::shared_ptr<elementwise_operation<aarch64>> elementwise_postop =
            nullptr,
        std::vector<std::map<std::string, int>> const&
            elementwise_postop_strides = {},
        std::optional<OptimizationConfiguration> const& optim_config =
            std::nullopt)
        : loop_nest_code_generator(
              arguments.get_order(), arguments.get_sizes(),
              arguments.get_C_axes(), arguments.get_A_axes(),
              arguments.get_B_axes(), arguments.get_C_strides(),
              arguments.get_A_strides(), arguments.get_B_strides(), op_pair,
              user_operation_unroll_limit, elementwise_preop,
              elementwise_preop_strides, elementwise_postop,
              elementwise_postop_strides, optim_config)
    {
    }
};

template <>
class loop_nest_code_generator<aarch64, true>
    : public loop_nest_fp16_code_generator<aarch64>
{
private:
    using impl_class = loop_nest_fp16_code_generator<aarch64>;

public:
    using impl_class::impl_class;
};

#    ifndef DABUN_HEADER_ONLY

extern template class loop_nest_code_generator<aarch64, false>;
extern template class loop_nest_code_generator<aarch64, true>;

#    endif

} // namespace arm
} // namespace dabun

#endif
