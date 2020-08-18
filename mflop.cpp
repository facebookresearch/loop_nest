#include "code_generator.h"
#include "math.h"

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

class test : public facebook::sysml::aot::code_generator<void(float*)>
{
public:
    test(int iterations)
    {
        // vmovups(ymm31, ptr[rdi]);

        Label loopLabel;
        mov(rax, 0);
        L(loopLabel);

        vbroadcastss(ymm14, ptr[rdi]);
        vbroadcastss(ymm15, ptr[rdi]);

        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 14; ++j)
            {
                vfmadd231ps(Ymm(j), ymm14, ymm15);
            }
        }

        add(rax, 1);
        cmp(rax, iterations);
        jl(loopLabel);
        ret();
    }
};

int main()
{
    int iterations = 10000000;

    auto fn = test(iterations).get_shared();
    fn.save_to_file("zi.asm");

    float pera[160] = {0.f};

    auto secs = measureFastestWithWarmup([&]() { fn(pera); }, 10, 100);

    double gflops = 1.0 * iterations * 6 * 14 * 16 / 1000000000;

    std::cout << (gflops / secs) << "\n";
    // std::cout << fn() << "\n";
}
