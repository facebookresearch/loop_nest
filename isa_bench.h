#pragma once

#include "code_generator.h"
#include "isa.h"
#include "math.h"
#include "utils.h"

namespace facebook
{
namespace sysml
{

template <class ISA>
struct gflops_benchmark;

#if defined(LOOP_NEST_ARM) || defined(ARM_LOOP_NEST)

template <>
struct gflops_benchmark<aot::aarch64>
{
private:
    static constexpr int vector_size =
        aot::isa_traits<aot::aarch64>::vector_size;

    class test : public aot::code_generator<void(int)>
    {
    private:
        Reg64 ZeroReg_ = x4;

    public:
        test()
        {
            eor(ZeroReg_, ZeroReg_, ZeroReg_);
            ins(v0.d[0], ZeroReg_);
            ins(v0.d[1], ZeroReg_);
            ins(v1.d[0], ZeroReg_);
            ins(v1.d[1], ZeroReg_);

            auto loopLabel = make_label();
            L_aarch64(*loopLabel);

            for (int i = 0; i < 10; ++i)
            {
                for (int r = 2; r < 8; ++r)
                {
                    fmla(VReg(r).s4, v0.s4, v1.s4);
                }
                for (int r = 16; r < 32; ++r)
                {
                    fmla(VReg(r).s4, v0.s4, v1.s4);
                }
            }

            sub(x0, x0, 1);
            cbnz(x0, *loopLabel);

            ret();
        }
    };

public:
    static double do_bench(int iterations = 10000000)
    {
        auto fn = test().get_shared();

        auto secs =
            measureFastestWithWarmup([&]() { fn(iterations); }, 10, 100);

        double gflops =
            2.0 * iterations * 10 * (16 + 6) * vector_size / 1000000000;

        return gflops / secs;
    }
};

#else

template <class ISA>
struct gflops_benchmark
{
private:
    static_assert(std::is_same_v<ISA, aot::avx2> ||
                  std::is_same_v<ISA, aot::avx512> ||
                  std::is_same_v<ISA, aot::avx2_plus>);

    using Vmm = std::conditional_t<std::is_same_v<ISA, aot::avx512>, Xbyak::Zmm,
                                   Xbyak::Ymm>;
    static constexpr int vector_size = aot::isa_traits<ISA>::vector_size;
    static constexpr int num_vector_regs =
        aot::isa_traits<ISA>::total_vector_registers;

    class test : public aot::code_generator<void(float*)>
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
        float data;

        auto secs = measureFastestWithWarmup([&]() { fn(&data); }, 10, 100);

        double gflops = 2.0 * iterations * 10 * (num_vector_regs - 2) *
                        vector_size / 1000000000;

        return gflops / secs;
    }
};

#endif

template <class T>
double peak_gflops(T const&, int iterations = 1000000)
{
    return gflops_benchmark<T>::do_bench(iterations);
}

} // namespace sysml
} // namespace facebook
