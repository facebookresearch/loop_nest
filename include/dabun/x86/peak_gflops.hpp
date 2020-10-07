#pragma once

#include "dabun/code_generator.hpp"
#include "dabun/isa.hpp"
#include "dabun/math.hpp"
#include "dabun/measure.hpp"

namespace dabun
{
namespace x86
{

template <class ISA>
struct bench_gflops
{
private:
    static_assert(std::is_same_v<ISA, avx2> || std::is_same_v<ISA, avx512> ||
                  std::is_same_v<ISA, avx2_plus>);

    using Vmm =
        std::conditional_t<std::is_same_v<ISA, avx512>, Xbyak::Zmm, Xbyak::Ymm>;
    static constexpr int vector_size = isa_traits<ISA>::vector_size;
    static constexpr int num_vector_regs =
        isa_traits<ISA>::total_vector_registers;

    class test : public code_generator<void(float*)>
    {
    public:
        test(int iterations)
        {
            Label loopLabel;
            mov(rax, 0);
            L(loopLabel);

            vbroadcastss(Vmm(num_vector_regs - 1), ptr[rdi]);
            vbroadcastss(Vmm(num_vector_regs - 2), ptr[rdi]);

            for (int i = 0; i < 10; ++i)
            {
                for (int j = 0; j < num_vector_regs - 2; ++j)
                {
                    vfmadd231ps(Vmm(j), Vmm(num_vector_regs - 1),
                                Vmm(num_vector_regs - 2));
                }
            }

            add(rax, 1);
            cmp(rax, iterations);
            jl(loopLabel);
            ret();
        }
    };

public:
    static double do_bench(int iterations = 10000000)
    {
        auto  fn = test(iterations).get_shared();
        float data[1] = {0};

        auto secs = measure_fastest([&]() { fn(data); }, 100);

        double gflops = 2.0 * iterations * 10 * (num_vector_regs - 2) *
                        vector_size / 1000000000;

        return gflops / secs;
    }
};

} // namespace x86
} // namespace dabun
