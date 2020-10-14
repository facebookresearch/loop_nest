#pragma once

#include "dabun/code_generator.hpp"
#include "dabun/isa.hpp"
#include "dabun/math.hpp"
#include "dabun/measure.hpp"

namespace dabun
{
namespace arm
{

template <class ISA>
struct bench_gflops;

template <>
struct bench_gflops<aarch64>
{
private:
    static constexpr int vector_size = isa_traits<aarch64>::vector_size;

    class test : public code_generator<void(int)>
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

        auto secs = measure_fastest([&]() { fn(iterations); }, 100);

        double gflops =
            2.0 * iterations * 10 * (16 + 6) * vector_size / 1000000000;

        return gflops / secs;
    }
};

} // namespace arm
} // namespace dabun
