#include "log.h"
#include "most_frequent_queue.h"
#include "multi_vmm.h"

#include "code_generator.h"
#include "math.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace facebook
{
namespace sysml
{
namespace aot
{

static constexpr int  vector_size = 8;
static constexpr bool DEBUG       = true;
static constexpr bool INFO        = true;

class FMA_loop_nest_jitter
    : public code_generator<void(float* C, float const* A, float const* B)>
{
private:
    using base = code_generator<void(float* C, float const* A, float const* B)>;
    using multi_zmm = multi_vmm<Ymm>;

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

    auto at_stack_offset(int off)
    {
        return ptr[rsp + (stack_offset - off) * 8];
    }

    auto at_stack_offset(Xbyak::AddressFrame const& af, int off)
    {
        return af[rsp + (stack_offset - off) * 8];
    }

private:
    Reg64 CReg_ = rdi;
    Reg64 AReg_ = rsi;
    Reg64 BReg_ = rdx;

    Reg64 loopReg_ = rcx;

    struct loop_descriptor
    {
        std::string var;
        int         begin;
        int         end;
        int         delta;
        int         iterations;
        int         tail;
    };

    struct memory_argument
    {
        std::string name;
        int         offset;
        int         len;
        Reg64       reg;

        bool operator<(memory_argument const& o) const
        {
            return std::tie(offset, name, len) <
                   std::tie(o.offset, o.name, o.len);
        }
        bool operator==(memory_argument const& o) const
        {
            return std::tie(offset, name, len) ==
                   std::tie(o.offset, o.name, o.len);
        }

        std::string readable() const
        {
            return name + "[" + std::to_string(offset) + ":" +
                   std::to_string(len) + "]";
        }
    };

    static void print_ld(loop_descriptor const& l)
    {
        LN_LOG(INFO) << "Loop over " << l.var << " from " << l.begin << " to "
                     << l.end << " by " << l.delta << " (total of "
                     << l.iterations << " iterations with " << l.tail << " tail"
                     << "\n";
    }

public:
    FMA_loop_nest_jitter(std::vector<std::pair<std::string, int>> const& order,
                         std::map<std::string, int> const&               sizes,
                         std::set<std::string> const&      C_formula,
                         std::set<std::string> const&      A_formula,
                         std::set<std::string> const&      B_formula,
                         std::map<std::string, int> const& C_strides,
                         std::map<std::string, int> const& A_strides,
                         std::map<std::string, int> const& B_strides)
    {
        // Check which ones are vectorized in the innermost loop
        bool is_C_vectorized = C_strides.count(order.back().first) == 1 &&
                               C_strides.at(order.back().first) == 1;
        bool is_A_vectorized = A_strides.count(order.back().first) == 1 &&
                               A_strides.at(order.back().first) == 1;
        bool is_B_vectorized = B_strides.count(order.back().first) == 1 &&
                               B_strides.at(order.back().first) == 1;

        int C_vector_len = is_C_vectorized ? vector_size : 1;

        LN_LOG(DEBUG) << "C_vector_len is: " << C_vector_len << "\n";

        // Well something has to be vectorized
        assert(is_C_vectorized || is_B_vectorized || is_A_vectorized);

        // This assures no gathers required, to be relaxed later
        assert((C_strides.count(order.back().first) == 0 ||
                C_strides.at(order.back().first) == 1) &&
               (B_strides.count(order.back().first) == 0 ||
                B_strides.at(order.back().first) == 1) &&
               (A_strides.count(order.back().first) == 0 ||
                A_strides.at(order.back().first) == 1));

        std::string vectorized_var = order.back().first;

        std::vector<loop_descriptor> loops;

        // Compute the ranges of all loops
        {
            auto sizes_copy = sizes;
            for (auto const& o : order)
            {
                // for now force no tail compute
                assert(sizes_copy[o.first] &&
                       (sizes_copy[o.first] % o.second == 0));
                loops.push_back({o.first, 0, sizes_copy[o.first], o.second,
                                 sizes_copy[o.first] / o.second,
                                 sizes_copy[o.first] % o.second});
                sizes_copy[o.first] = o.second;

                print_ld(loops.back());
            }
        }

        // Get the total number of vector registers required to store full
        // result.

        int registers_required =
            std::accumulate(sizes.begin(), sizes.end(), 1,
                            [&](int v, auto const& s) {
                                return C_formula.count(s.first) ? v * s.second
                                                                : v;
                            }) /
            C_vector_len;

        LN_LOG(DEBUG) << "C VECTOR LEN: " << C_vector_len << "\n";

        LN_LOG(DEBUG) << "REGISTERS REQUIRED FOR C: " << registers_required
                      << "\n";

        int first_loop_that_can_hold_C = 0;

        // Find the first loop that can hold C in the register file
        for (; first_loop_that_can_hold_C < loops.size();
             ++first_loop_that_can_hold_C)
        {
            // this gives only one two extra register for other values
            // should be fine for noe
            if (registers_required <= 14)
            {
                break;
            }

            auto const& loop = loops[first_loop_that_can_hold_C];

            if (C_formula.count(loop.var))
            {
                int reduction = loop.end / loop.delta;
                registers_required /= reduction;
            }

            std::cout << "AT LOOP: " << first_loop_that_can_hold_C
                      << " REQ: " << registers_required << "\n";
        }

        LN_LOG(DEBUG) << "CAN HOLD C IN REGISTER FILE AT LOOP: "
                      << first_loop_that_can_hold_C << " USING "
                      << registers_required << " REGISTERS\n";

        // Unrolling is happening below the first loop that can hold C in
        // registers

        int total_fma_operations =
            std::accumulate(
                sizes.begin(), sizes.end(), 1,
                [&](int v, auto const& s) { return v * s.second; }) /
            vector_size;

        LN_LOG(DEBUG) << "REQUIRED FMA OPERATIONS: " << total_fma_operations
                      << "\n";

        // Here we put some fake unroll limit
        static constexpr int max_fmas_unrolled = 512;

        int unroll_stage = 0;

        for (; unroll_stage < first_loop_that_can_hold_C; ++unroll_stage)
        {
            auto const& loop = loops[unroll_stage];

            int reduction = loop.end / loop.delta;
            total_fma_operations /= reduction;
        }

        LN_LOG(DEBUG) << "REQUIRED FMA BELOW C IN REGISTERS: "
                      << total_fma_operations << "\n";

        while (total_fma_operations > max_fmas_unrolled)
        {
            auto const& loop      = loops[unroll_stage];
            int         reduction = loop.end / loop.delta;
            total_fma_operations /= reduction;
            ++unroll_stage;
        }

        LN_LOG(DEBUG) << "UNROLL STAGE MOVED TO: " << unroll_stage
                      << " REQUIRES: " << total_fma_operations << " FMAs\n";

        int inner_fma_operations = total_fma_operations;

        if (first_loop_that_can_hold_C < unroll_stage)
        {
            first_loop_that_can_hold_C = unroll_stage;
            while (first_loop_that_can_hold_C > 0 &&
                   C_formula.count(loops[first_loop_that_can_hold_C - 1].var) ==
                       0)
            {
                --first_loop_that_can_hold_C;
                auto const& loop      = loops[first_loop_that_can_hold_C];
                int         expansion = loop.end / loop.delta;
                inner_fma_operations *= expansion;
            }

            LN_LOG(DEBUG) << "LOAD/STORE C MOVED TO LOOP: "
                          << first_loop_that_can_hold_C << " OVER "
                          << loops[first_loop_that_can_hold_C].var << " WITH "
                          << inner_fma_operations << " INNER FMAs\n";
        }

        int loop_that_holds_C = first_loop_that_can_hold_C;

        // We have so far determined two important loops in the nest
        //
        // 1) The loop where the C elements will be loaded to registers as
        //    well as stored back to memroy
        // 2) The loop which will be fully unrolled together with all nested
        //    loops

        // Now we are going to collect all the loads/stores required
        // for the nest starting at loop_that_can_hold_C

        std::map<std::string, int> coordinates;

        auto get_location = [&](std::map<std::string, int> const& strides) {
            int r = 0;
            for (auto const& s : strides)
            {
                r += coordinates[s.first] * s.second;
            }
            return r;
        };

        std::set<memory_argument>            collected_load_store;
        std::map<memory_argument, Ymm>       CZmms;
        std::map<memory_argument, multi_zmm> CMultiZmms;

        {
            std::function<void(int)> collect = [&](int depth) {
                auto const& loop = loops[depth];

                if (depth == order.size() - 1)
                {
                    assert(loop.delta == 1);
                    assert(loop.iterations % vector_size == 0);

                    // assert(C_formula.count(loop.var) == 1);

                    int save = coordinates[loop.var];
                    for (int i = 0; i < loop.end; i += vector_size)
                    {
                        auto loc        = get_location(C_strides);
                        auto load_store = memory_argument{
                            "C", get_location(C_strides), C_vector_len, CReg_};
                        collected_load_store.insert(load_store);
                        coordinates[loop.var] += vector_size;
                    }
                    coordinates[loop.var] = save;
                }
                else
                {
                    int save = coordinates[loop.var];
                    for (int i = 0; i < loop.iterations; ++i)
                    {
                        collect(depth + 1);
                        coordinates[loop.var] += loop.delta;
                    }
                    coordinates[loop.var] = save;
                }
            };

            collect(loop_that_holds_C);

            int next = 2;

            // TODO(zi) better heuristics here
            if (collected_load_store.size() < 8 && inner_fma_operations > 1000)
            {
                int per_register = 8 / collected_load_store.size();
                for (auto const& c : collected_load_store)
                {
                    std::cout << "LOAD/STORE: " << c.readable() << "\n";
                    CMultiZmms[c] = multi_zmm(per_register, next);
                    CZmms[c]      = CMultiZmms[c].current();
                    next += per_register;
                }
            }
            else
            {
                for (auto const& c : collected_load_store)
                {
                    std::cout << "LOAD/STORE: " << c.readable() << "\n";
                    CZmms[c] = Ymm(next++);
                }
            }

            assert(next <= 32);
        }

        std::vector<std::string> tabs = {""};

        // Limits per nested partition of the variable
        std::map<std::string, std::vector<int>> limits;
        for (auto const& p : sizes)
        {
            limits[p.first].push_back(p.second);
        }

        auto push_ptrs = [&](std::string const& var) {
            if (C_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "PUSH C_ptr\n";
                push(CReg_);
            }
            if (B_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "PUSH B_ptr\n";
                push(BReg_);
            }
            if (A_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "PUSH A_ptr\n";
                push(AReg_);
            }
        };

        auto pop_ptrs = [&](std::string const& var) {
            if (A_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "POP A_ptr\n";
                pop(AReg_);
            }
            if (B_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "POP B_ptr\n";
                pop(BReg_);
            }
            if (C_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "POP C_ptr\n";
                pop(CReg_);
            }
        };

        auto update_ptrs = [&](std::string const& var, int delta) {
            if (A_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "A_ptr += " << delta << " * "
                             << A_strides.at(var) << "\n";
                add(AReg_, A_strides.at(var) * delta * 4);
            }
            if (B_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "B_ptr += " << delta << " * "
                             << B_strides.at(var) << "\n";
                add(BReg_, B_strides.at(var) * delta * 4);
            }
            if (C_strides.count(var))
            {
                LN_LOG(INFO) << tabs.back() << "C_ptr += " << delta << " * "
                             << C_strides.at(var) << "\n";
                add(CReg_, C_strides.at(var) * delta * 4);
            }
        };

        coordinates.clear();

        struct fma_op
        {
            memory_argument dest, src1, src2;
        };

        std::vector<fma_op>   unrolled_fmas;
        std::set<std::string> unrolled_dimensions;

        std::function<void(int)> prepare_vectorized_loop = [&](int depth) {
            // only along last dimension

            assert(depth == order.size() - 1);

            auto const& loop = loops[depth];

            unrolled_dimensions.insert(loop.var);

            assert(loop.delta == 1);
            assert(C_strides.count(loop.var) == 0 ||
                   C_strides.at(loop.var) == 1);
            assert(A_strides.count(loop.var) == 0 ||
                   A_strides.at(loop.var) == 1);
            assert(B_strides.count(loop.var) == 0 ||
                   B_strides.at(loop.var) == 1);

            int save = coordinates[loop.var];
            for (int i = 0; i < loop.end; i += vector_size)
            {
                memory_argument dest{"C", get_location(C_strides), C_vector_len,
                                     CReg_};
                memory_argument src1{
                    "B", get_location(B_strides),
                    (B_strides.count(loop.var) ? vector_size : 1), BReg_};

                memory_argument src2{
                    "A", get_location(A_strides),
                    (A_strides.count(loop.var) ? vector_size : 1), AReg_};

                unrolled_fmas.push_back({dest, src1, src2});
                coordinates[loop.var] += vector_size;
            }
            coordinates[loop.var] = save;
        };

        std::function<void(int)> prepare_unrolled_fmas = [&](int depth) {
            if (depth + 1 < order.size())
            {
                auto const& loop = loops[depth];

                unrolled_dimensions.insert(loop.var);

                int save = coordinates[loop.var];
                for (int i = 0; i < loop.end; i += loop.delta)
                {
                    prepare_unrolled_fmas(depth + 1);
                    coordinates[loop.var] += loop.delta;
                }
                coordinates[loop.var] = save;
            }
            else
            {
                prepare_vectorized_loop(depth);
            }
        };

        auto issue_unrolled_fmas = [&]() {
            most_frequent_queue<memory_argument> queue;

            for (auto const& inst : unrolled_fmas)
            {
                queue.inc(inst.src1);
                queue.inc(inst.src2);
            }

            while (queue.size() > 0)
            {
                auto addr = queue.top();
                queue.pop();

                LN_LOG(INFO)
                    << tabs.back() << "LOAD " << addr.readable() << "\n";

                // TODO(zi) make sure not to issue any unused loads

                if (addr.len == vector_size)
                {
                    vmovups(ymm0, ptr[addr.reg + addr.offset * 4]);
                }
                else
                {
                    vbroadcastss(ymm0, ptr[addr.reg + addr.offset * 4]);
                }

                for (auto it = unrolled_fmas.begin();
                     it != unrolled_fmas.end();)
                {
                    if (it->src1 == addr || it->src2 == addr)
                    {
                        // vfmadd231ps
                        auto src1 = it->src1;
                        auto src2 = it->src2;
                        if (addr == it->src2)
                        {
                            std::swap(src1, src2);
                        }

                        queue.dec(src2);

                        auto arg2_ptr = ptr[src2.reg + src2.offset * 4];

                        if (CMultiZmms.size())
                        {
                            if (src2.len == 1)
                            {
                                vbroadcastss(ymm1, arg2_ptr);
                                vfmadd231ps(CMultiZmms[it->dest]++, ymm0, ymm1);
                            }
                            else
                            {
                                vfmadd231ps(CMultiZmms[it->dest]++, ymm0,
                                            arg2_ptr);
                            }
                        }
                        else
                        {

                            if (src2.len == 1)
                            {
                                vbroadcastss(ymm1, arg2_ptr);
                                vfmadd231ps(CZmms[it->dest], ymm0, ymm1);
                            }
                            else
                            {
                                vfmadd231ps(CZmms[it->dest], ymm0, arg2_ptr);
                            }
                        }

                        LN_LOG(INFO) << tabs.back() << it->dest.readable()
                                     << " += " << it->src1.readable() << " * "
                                     << it->src2.readable() << "\n";
                        it = unrolled_fmas.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }
        };

        auto analyze_unrolled_fmas = [&]() {
            most_frequent_queue<memory_argument> queue;

            auto unrolled_fmas_copy = unrolled_fmas;

            for (auto const& inst : unrolled_fmas)
            {
                queue.inc(inst.src1);
                queue.inc(inst.src2);
            }

            while (queue.size() > 0)
            {
                auto addr = queue.top();
                queue.pop();

                std::vector<int> addresses;

                for (auto it = unrolled_fmas_copy.begin();
                     it != unrolled_fmas_copy.end();)
                {
                    if (it->src1 == addr || it->src2 == addr)
                    {
                        // vfmadd231ps
                        auto src1 = it->src1;
                        auto src2 = it->src2;
                        if (addr == it->src2)
                        {
                            std::swap(src1, src2);
                        }

                        queue.dec(src2);

                        addresses.push_back(src2.offset * 4);

                        it = unrolled_fmas_copy.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }

                // Find patterns
                std::sort(addresses.begin(), addresses.end());
                std::map<int, int> patterns;

                for (std::size_t i = 1; i < addresses.size(); ++i)
                {
                    auto diff = addresses[i] - addresses[0];
                    auto it   = std::find_if(
                        patterns.begin(), patterns.end(),
                        [&](auto const& p) { return (diff % p.first) == 0; });
                    if (it != patterns.end())
                    {
                        ++it->second;
                    }
                    else
                    {
                        patterns[diff] = 1;
                    }
                }

                for (auto const& p : patterns)
                {
                    LN_LOG(INFO)
                        << "PATTERN: " << p.first << " of " << p.second << "\n";
                }
            }
        };

        prepare_unrolled_fmas(unroll_stage);

        assert(unrolled_fmas.size() == total_fma_operations);

        analyze_unrolled_fmas();

        LN_LOG(DEBUG) << "Unrolled dimensions: ";
        for (auto const& d : unrolled_dimensions)
        {
            LN_LOG(DEBUG) << d << " ";
        }
        LN_LOG(DEBUG) << "\n";

        // TODO(zi) Analyze the unrolled FMAs for address compression

        std::function<void(int)> issue_loop = [&](int depth) {
            LN_LOG(INFO) << tabs.back() << "// DEPTH: " << depth << "\n";

            if (depth == loop_that_holds_C)
            {
                // issue loads
                for (auto const& c : collected_load_store)
                {
                    LN_LOG(INFO)
                        << tabs.back() << "LOAD " << c.readable() << "\n";
                    if (C_vector_len == vector_size)
                    {
                        vmovups(CZmms[c], ptr[CReg_ + c.offset * 4]);
                        if (CMultiZmms.size())
                        {
                            ++CMultiZmms[c];
                            for (int s = 1; s < CMultiZmms[c].size(); ++s)
                            {
                                vxorpd(CMultiZmms[c].current(),
                                       CMultiZmms[c].current(),
                                       CMultiZmms[c].current());
                                ++CMultiZmms[c];
                            }
                        }
                    }
                    else
                    {
                        vxorpd(CZmms[c], CZmms[c], CZmms[c]);
                        vmovss(Xmm(CZmms[c].getIdx()),
                               ptr[CReg_ + c.offset * 4]);

                        if (CMultiZmms.size())
                        {
                            ++CMultiZmms[c];
                            for (int s = 1; s < CMultiZmms[c].size(); ++s)
                            {
                                vxorpd(CMultiZmms[c].current(),
                                       CMultiZmms[c].current(),
                                       CMultiZmms[c].current());
                                ++CMultiZmms[c];
                            }
                        }
                    }
                }
            }

            if (depth == unroll_stage)
            {
                issue_unrolled_fmas();
            }
            else if (loops[depth].iterations > 0) // this is kind of guaranteed
            {
                auto const& loop = loops[depth];

                std::string var_name =
                    loop.var + "_" + std::to_string(loop.delta);

                push_ptrs(loop.var);
                push(loopReg_);
                xor_(loopReg_, loopReg_);

                Label loopLabel;
                L(loopLabel);

                LN_LOG(INFO)
                    << tabs.back() << "FOR: " << var_name << " FROM 0 TO "
                    << loop.end << " BY " << loop.delta << " {\n";

                tabs.push_back(tabs.back() + "    ");

                issue_loop(depth + 1);

                update_ptrs(loop.var, loop.delta);

                add(loopReg_, 1);
                cmp(loopReg_, loop.iterations);
                jl(loopLabel);

                tabs.pop_back();
                LN_LOG(INFO) << tabs.back() << "} END FOR\n";

                pop(loopReg_);
                pop_ptrs(loop.var);
            }

            if (depth == loop_that_holds_C)
            {
                for (auto const& c : collected_load_store)
                {
                    LN_LOG(INFO)
                        << tabs.back() << "STORE " << c.readable() << "\n";

                    if (CMultiZmms.count(c))
                    {
                        CMultiZmms[c].reduce(*this);
                    }

                    if (C_vector_len == vector_size)
                    {
                        vmovups(ptr[CReg_ + c.offset * 4], CZmms[c]);
                    }
                    else
                    {
                        // Needs the horizontal sum
                        // vextractf64x4(ymm1, CZmms[c], 1);
                        // vaddps(ymm1, ymm1, CZmms[c]);

                        vextractf128(xmm0, CZmms[c], 1);
                        vaddps(xmm0, xmm0, CZmms[c]);

                        // xmm1 = xmm0[1,0]
                        vpermilpd(xmm1, xmm0, 1);
                        vaddps(xmm0, xmm0, xmm1);
                        // xmm1 = xmm0[1,0,3,2]
                        vpermilps(xmm1, xmm0, 177);
                        vaddps(xmm0, xmm0, xmm1);

                        vmovss(ptr[CReg_ + c.offset * 4], xmm0);
                    }
                }
            }
        };

        issue_loop(0);

        // This is apparently very important as it can slow down
        // legacy SSE code upon return.
        // software.intel.com/en-us/forums/intel-isa-extensions/topic/704023
        vzeroupper();
        ret();
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook

void baseline_MM(unsigned ArCr, unsigned AcBr, unsigned BcCc, int LDA, int LDB,
                 int LDC, float const* AData, float const* BData, float* CData)
{
    for (int arcr = 0; arcr < ArCr; ++arcr)
    {
        for (int bccc = 0; bccc < BcCc; ++bccc)
        {
            CData[arcr * LDC + bccc] = 0.f;
            for (int i = 0; i < AcBr; ++i)
            {
                CData[arcr * LDC + bccc] +=
                    AData[arcr * LDA + i] * BData[i * LDB + bccc];
            }
        }
    }
}

void baseline_MM_row_col_major(unsigned ArCr, unsigned AcBr, unsigned BcCc,
                               int LDA, int LDB, int LDC, float const* AData,
                               float const* BData, float* CData)
{
    for (int arcr = 0; arcr < ArCr; ++arcr)
    {
        for (int bccc = 0; bccc < BcCc; ++bccc)
        {
            CData[arcr * LDC + bccc] = 0.f;
            for (int i = 0; i < AcBr; ++i)
            {
                CData[arcr * LDC + bccc] +=
                    AData[arcr * LDA + i] * BData[i + bccc * LDB];
            }
        }
    }
}

void baseline_Conv(unsigned COUT, unsigned CIN, unsigned OH, int OW, int KH,
                   int KW, float const* AData, float const* BData, float* CData)
{
    int IH = OH + KH - 1;
    int IW = OW + KW - 1;
    for (int cout = 0; cout < COUT; ++cout)
    {
        for (int oh = 0; oh < OH; ++oh)
        {
            for (int ow = 0; ow < OW; ++ow)
            {
                CData[cout + ow * COUT + oh * COUT * OW] = 0.f;
                for (int cin = 0; cin < CIN; ++cin)
                {
                    for (int kh = 0; kh < KH; ++kh)
                    {
                        for (int kw = 0; kw < KW; ++kw)
                        {
                            CData[cout + ow * COUT + oh * COUT * OW] +=
                                AData[cin + (oh + kh) * CIN * IW +
                                      (ow + kw) * CIN] *
                                BData[cout + cin * COUT + kw * CIN * COUT +
                                      kh * CIN * COUT * KW];
                        }
                    }
                }
            }
        }
    }
}

void baseline_Conv_NCHW8c(unsigned GOUT, unsigned COUT, unsigned GIN,
                          unsigned CIN, unsigned OH, int OW, int KH, int KW,
                          float const* AData, float const* BData, float* CData)
{
    int IH = OH + KH - 1;
    int IW = OW + KW - 1;
    for (int gout = 0; gout < GOUT; ++gout)
    {
        for (int cout = 0; cout < COUT; ++cout)
        {
            for (int oh = 0; oh < OH; ++oh)
            {
                for (int ow = 0; ow < OW; ++ow)
                {
                    // C[gout][h][w][cout]
                    CData[((gout * OH + oh) * OW + ow) * COUT + cout] = 0.f;
                    for (int gin = 0; gin < GIN; ++gin)
                    {
                        for (int cin = 0; cin < CIN; ++cin)
                        {
                            for (int kh = 0; kh < KH; ++kh)
                            {
                                for (int kw = 0; kw < KW; ++kw)
                                {
                                    CData[((gout * OH + oh) * OW + ow) * COUT +
                                          cout] +=
                                        AData[((gin * IH + (oh + kh)) * IW +
                                               (ow + kw)) *
                                                  CIN +
                                              cin] *
                                        // B[gin][gout][cin][kh][kw][cout]
                                        BData[((((gin * GOUT + gout) * CIN +
                                                 cin) *
                                                    KH +
                                                kh) *
                                                   KW +
                                               kw) *
                                                  COUT +
                                              cout];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <class Float>
Float maxAbsDiff(Float const* LBegin, Float const* LEnd, Float const* RBegin)
{
    Float res = 0;
    for (; LBegin != LEnd; ++LBegin, ++RBegin)
    {
        res = std::max(res, std::abs(*LBegin - *RBegin));
    }
    return res;
}

template <class Float>
std::vector<Float> getRandomVector(unsigned size)
{
    std::vector<Float> res(size);

    std::random_device rd;
    std::mt19937       gen(345);

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (auto& f : res)
    {
        f = dis(gen);
    }

    return res;
}

template <class Fn>
double measureFastestWithWarmup(Fn&& fn, int warmupIterations,
                                int measuredIterations = 1)
{
    for (int i = 0; i < warmupIterations; ++i)
    {
        fn();
    }

    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    auto nsecs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    for (int i = 1; i < measuredIterations; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        fn();
        end = std::chrono::high_resolution_clock::now();

        auto new_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();

        // LN_LOG(INFO) << "T: " << new_time << "\n";
        nsecs = std::min(nsecs, new_time);
    }

    return static_cast<double>(nsecs) / 1e9;
}

template <class BaseLineImpl, class JITImpl>
void check_correctness(BaseLineImpl&& baseline_fn, JITImpl&& jit_fn, int A_size,
                       int B_size, int C_size)
{
    auto A = getRandomVector<float>(A_size);
    auto B = getRandomVector<float>(B_size);

    auto CN = std::vector<float>(C_size);
    auto CJ = std::vector<float>(C_size);

    baseline_fn(CN.data(), A.data(), B.data());
    jit_fn(CJ.data(), A.data(), B.data());

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(CJ.data(), CJ.data() + C_size, CN.data()) << "\n";
}

template <class Fn>
void bench_implementation(Fn&& fn, int A_size, int B_size, int C_size,
                          double gflops, int warmup = 5, int iters = 10)
{
    auto A = getRandomVector<float>(A_size);
    auto B = getRandomVector<float>(B_size);
    auto C = std::vector<float>(C_size);

    auto secs = measureFastestWithWarmup(
        [&]() { fn(C.data(), A.data(), B.data()); }, warmup, iters);

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
}

int main()
{

    // (row)Vector-(row-major)Matrix product
    // C(c) = A(k) * B(k, c)
    // if (0)
    {
        int ArCr = 1;
        int AcBr = 64 * 128;
        int BcCc = 16;

        int k = AcBr;
        int c = BcCc;

        auto fn = facebook::sysml::aot::FMA_loop_nest_jitter(
                      {{"k", 64}, //
                       {"k", 1},  //
                       {"c", 1}}, //
                      {{"k", k}, {"c", c}},
                      // Vars of C (other variables are reduction variables)
                      {"c"},
                      // Variables of A
                      {"k"},
                      // Variables of B
                      {"c", "k"},
                      // C's strides for each variable
                      {{"c", 1}},
                      // A's strides for each variable
                      {{"k", 1}},
                      // B's strides for each variable
                      {{"k", c}, {"c", 1}})
                      .get_shared();

        fn.save_to_file("zi.asm");

        check_correctness(
            [=](float* Cdata, float const* Adata, float const* Bdata) {
                baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, Adata, Bdata,
                            Cdata);
            },
            fn, k, k * c, c);

        bench_implementation(fn, k, k * c, c, 1.0 * k * c * 2 / 1000000000, 10,
                             10000);
    }

    //return 0;

    // (row)Vector-(row-major)Matrix product
    // C(c) = A(k) * B(k, c)
    // if (0)
    {
        int ArCr = 1;
        int AcBr = 64;
        int BcCc = 8 * 14;

        int k = AcBr; // = 64
        int c = BcCc; // = 16 * 28

        auto fn = facebook::sysml::aot::FMA_loop_nest_jitter(
                      {{"k", 4},  //
                       {"k", 1},  //
                       {"c", 1}}, //
                      {{"k", k}, {"c", c}},
                      // Vars of C (other variables are reduction variables)
                      {"c"},
                      // Variables of A
                      {"k"},
                      // Variables of B
                      {"c", "k"},
                      // C's strides for each variable
                      {{"c", 1}},
                      // A's strides for each variable
                      {{"k", 1}},
                      // B's strides for each variable
                      {{"k", c}, {"c", 1}})
                      .get_shared();

        check_correctness(
            [=](float* Cdata, float const* Adata, float const* Bdata) {
                baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, Adata, Bdata,
                            Cdata);
            },
            fn, AcBr * ArCr, AcBr * BcCc, ArCr * BcCc);

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CJ = std::vector<float>(ArCr * BcCc);

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data()); }, 10, 1000);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // return 0;

    // (row-major)Matrix-(column)Vector product (requires horizontal sum)
    // C(r) = A(r, k) * B(k)
    // if (0)
    {
        int ArCr = 256;
        int AcBr = 256;
        int BcCc = 1;

        int k = AcBr;
        int r = ArCr;

        auto fn = facebook::sysml::aot::FMA_loop_nest_jitter(
                      {{"r", 16}, //
                       {"r", 1},  //
                       {"k", 64},
                       {"k", 1}}, //
                      {{"k", k}, {"r", r}},
                      // Vars of C (other variables are reduction variables)
                      {"r"},
                      // Variables of A
                      {"r", "k"},
                      // Variables of B
                      {"k"},
                      // C's strides for each variable
                      {{"r", 1}},
                      // A's strides for each variable
                      {{"r", k}, {"k", 1}},
                      // B's strides for each variable
                      {{"k", 1}})
                      .get_shared();

        fn.save_to_file("zi.asm");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = std::vector<float>(ArCr * BcCc);
        auto CJ = std::vector<float>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data()); }, 10, 1000);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // 2D convolution on NCHW16c layout example:
    // O(g_out, c_out, o_h, o_w) = I(g_in, c_in, o_h + k_h, ow + k_w) *
    //                             K(g_in, g_out, c_in, c_out, k_h, k_w)
    // if (0)
    {
        int GIN  = 128 / 16;
        int CIN  = 16;
        int GOUT = 128 / 16;
        int COUT = 16;
        int OS   = 56;
        int KS   = 3;
        int IS   = OS + KS - 1;

        auto fn = facebook::sysml::aot::FMA_loop_nest_jitter(
                      {{"g_out", 1}, //
                       {"o_w", 28},
                       {"o_h", 1},
                       {"g_in", 1},
                       {"c_in", 1},
                       {"o_w", 1}, //
                       //{"o_w", 1},    //
                       {"k_h", 1},    //
                       {"k_w", 1},    //
                       {"c_out", 1}}, //
                      // The second argument is a map of the dimension sizes
                      {{"g_out", GOUT},
                       {"c_out", COUT},
                       {"o_w", OS},
                       {"k_w", KS},
                       {"g_in", GIN},
                       {"c_in", CIN},
                       {"o_h", OS},
                       {"k_h", KS}},
                      // Vars of C (other variables are reduction variables)
                      {"g_out", "c_out", "o_w", "o_h"},
                      // Variables of A, note that i_w and i_h are not used
                      {"g_in", "c_in", "i_w", "i_h"},
                      // Variables of B
                      {"g_out", "g_in", "c_in", "c_out", "k_w", "k_h"},
                      // C's strides for each variable
                      {{"g_out", OS * OS * COUT},
                       {"o_h", OS * COUT},
                       {"o_w", COUT},
                       {"c_out", 1}},
                      // A's strides for each variable Note how we
                      // provide strides for i/k_h and i/k_w, this is
                      // because the access to A is based on output
                      // and reduction variables
                      {{"g_in", IS * IS * CIN},
                       {"o_h", IS * CIN},
                       {"k_h", IS * CIN},
                       {"o_w", CIN},
                       {"k_w", CIN},
                       {"c_in", 1}},
                      // B's strides for each variable
                      {{"g_in", COUT * KS * KS * CIN * GOUT},
                       {"g_out", COUT * KS * KS * CIN},
                       {"c_in", COUT * KS * KS},
                       {"k_h", COUT * KS},
                       {"k_w", COUT},
                       {"c_out", 1}})
                      .get_shared();

        fn.save_to_file("zi.asm");

        auto A  = getRandomVector<float>(GIN * CIN * IS * IS);
        auto B  = getRandomVector<float>(GOUT * GIN * COUT * CIN * KS * KS);
        auto CN = std::vector<float>(GOUT * COUT * OS * OS);
        auto CJ = std::vector<float>(GOUT * COUT * OS * OS);

        baseline_Conv_NCHW8c(GOUT, COUT, GIN, CIN, OS, OS, KS, KS, A.data(),
                             B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + COUT * OS * OS,
                                CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data()); }, 10, 50);

        double gflops =
            2.0 * GIN * GOUT * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // 2D convolution example:
    // O(c_out, o_h, o_w) = I(c_i, o_h + k_h, ow + k_w) * K(c_o, c_i, k_h, k_w)
    // if (0)
    {
        int CIN  = 128;
        int COUT = 128;
        int OS   = 56;
        int KS   = 3;
        int IS   = OS + KS - 1;
        ;

        auto fn = facebook::sysml::aot::FMA_loop_nest_jitter(
                      {{"c_out", 16}, //
                       {"o_h", 1},
                       {"o_w", 28},
                       {"c_in", 16},
                       {"c_in", 1},
                       {"o_w", 1}, //
                       //{"o_w", 1},    //
                       {"k_h", 1},    //
                       {"k_w", 1},    //
                       {"c_out", 1}}, //
                      // The second argument is a map of the dimension sizes
                      {{"c_out", COUT},
                       {"o_w", OS},
                       {"k_w", KS},
                       {"c_in", CIN},
                       {"o_h", OS},
                       {"k_h", KS}},
                      // Vars of C (other variables are reduction variables)
                      {"c_out", "o_w", "o_h"},
                      // Variables of A, note that i_w and i_h are not used
                      {"c_in", "i_w", "i_h"},
                      // Variables of B
                      {"c_in", "c_out", "k_w", "k_h"},
                      // C's strides for each variable
                      {{"o_w", COUT}, {"c_out", 1}, {"o_h", COUT * OS}},
                      // A's strides for each variable Note how we
                      // provide strides for i/k_h and i/k_w, this is
                      // because the access to A is based on output
                      // and reduction variables
                      {{"o_w", CIN},
                       {"k_w", CIN},
                       {"c_in", 1},
                       {"o_h", IS * CIN},
                       {"k_h", IS * CIN}},
                      // B's strides for each variable
                      {{"c_out", 1},
                       {"c_in", COUT},
                       {"k_w", COUT * CIN},
                       {"k_h", COUT * CIN * KS}})
                      .get_shared();

        fn.save_to_file("zi.asm");

        auto A  = getRandomVector<float>(CIN * IS * IS);
        auto B  = getRandomVector<float>(COUT * CIN * KS * KS);
        auto CN = std::vector<float>(COUT * OS * OS);
        auto CJ = std::vector<float>(COUT * OS * OS);

        baseline_Conv(COUT, CIN, OS, OS, KS, KS, A.data(), B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + COUT * OS * OS,
                                CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data()); }, 10, 50);

        double gflops = 2.0 * CIN * COUT * OS * OS * KS * KS / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // Matrix-Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 120 * 32;
        int AcBr = 256;
        int BcCc = 256;

        auto fn = facebook::sysml::aot::FMA_loop_nest_jitter(
                      // The first argument is the loop order in the form of
                      // {dimension, stride}.  For now the outer dimension has
                      // to divide the stride.  This is effectively the same as
                      // Halide's split into outer and inner variable, but can
                      // have arbitray number of splits.
                      {{"AcBr", 256}, // To block B in L2 cache
                       {"ArCr", 10},  // This and the next are for the register
                                      // blocking of C - 30 vector registers of
                                      // each holding 16 values
                       {"BcCc", 16},
                       {"AcBr", 4}, // broken up to allow for unrolling of 4
                       {"AcBr", 1}, // inner loops, should handle differently
                                    // later
                       {"ArCr", 1},
                       {"BcCc", 1}},
                      // The second argument is a map of the dimension sizes
                      {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
                      // Vars of C (other variables are reduction variables)
                      {"ArCr", "BcCc"},
                      // Variables of A
                      {"ArCr", "AcBr"},
                      // Variables of B
                      {"AcBr", "BcCc"},
                      // C's strides for each variable.  Note that the strides
                      // data is a superset of the previous argument (variables
                      // of C).  I'm still deciding on the final design,
                      // possibly allowing for null strides that will just
                      // deduce them from the sizes, or some special structs
                      // indicating the layout (ie row-major, col-major).  In
                      // this case the vars have to be ordered though...
                      // Many decisions to make...
                      {{"ArCr", BcCc}, {"BcCc", 1}},
                      // A's strides for each variable
                      {{"ArCr", AcBr}, {"AcBr", 1}},
                      // B's strides for each variable
                      {{"AcBr", BcCc}, {"BcCc", 1}})
                      .get_shared();

        fn.save_to_file("zi.asm");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = std::vector<float>(ArCr * BcCc);
        auto CJ = std::vector<float>(ArCr * BcCc);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, BcCc, BcCc, A.data(), B.data(),
                    CN.data());

        fn(CJ.data(), A.data(), B.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data()); }, 10, 1000);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }

    // (row-major)Matrix-(col-major)Matrix product
    // C(r, c) = A(r, k) * B(k, c)
    // if (0)
    {
        int ArCr = 120 * 32;
        int AcBr = 256;
        int BcCc = 256;

        auto fn = facebook::sysml::aot::FMA_loop_nest_jitter(
                      // The first argument is the loop order in the form of
                      // {dimension, stride}.  For now the outer dimension has
                      // to divide the stride.  This is effectively the same as
                      // Halide's split into outer and inner variable, but can
                      // have arbitray number of splits.
                      {{"ArCr", 16}, // This and the next are for the register
                                     // blocking of C - 30 vector registers of
                                     // each holding 16 values
                       {"BcCc", 16},
                       {"ArCr", 1},
                       {"BcCc", 1},
                       {"AcBr", 1}},
                      // The second argument is a map of the dimension sizes
                      {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
                      // Vars of C (other variables are reduction variables)
                      {"ArCr", "BcCc"},
                      // Variables of A
                      {"ArCr", "AcBr"},
                      // Variables of B
                      {"AcBr", "BcCc"},
                      // C's strides for each variable.  Note that the strides
                      // data is a superset of the previous argument (variables
                      // of C).  I'm still deciding on the final design,
                      // possibly allowing for null strides that will just
                      // deduce them from the sizes, or some special structs
                      // indicating the layout (ie row-major, col-major).  In
                      // this case the vars have to be ordered though...
                      // Many decisions to make...
                      {{"ArCr", BcCc}, {"BcCc", 1}},
                      // A's strides for each variable
                      {{"ArCr", AcBr}, {"AcBr", 1}},
                      // B's strides for each variable
                      {{"AcBr", 1}, {"BcCc", AcBr}})
                      .get_shared();

        fn.save_to_file("zi.asm");

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = std::vector<float>(ArCr * BcCc);
        auto CJ = std::vector<float>(ArCr * BcCc);

        baseline_MM_row_col_major(ArCr, AcBr, BcCc, BcCc, BcCc, AcBr, A.data(),
                                  B.data(), CN.data());

        fn(CJ.data(), A.data(), B.data());

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";

        auto secs = measureFastestWithWarmup(
            [&]() { fn(CJ.data(), A.data(), B.data()); }, 10, 1000);

        double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

        std::cout << "GFLOPS: " << (gflops / secs) << "\n";
    }
}
