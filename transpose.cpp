#include "transposer.h"
#include "transposer_baseline.h"
#include "transposer_bench.h"
#include "utils.h"

int main()
{

    using facebook::sysml::aot::aarch64;
    using facebook::sysml::aot::aot_fn_cast;
    using facebook::sysml::aot::avx2;
    using facebook::sysml::aot::avx2_plus;
    using facebook::sysml::aot::avx512;

    int R = 2;
    int C = 2;

    auto A  = getRandomVector<float>(R * C);
    auto B  = getRandomVector<float>(R * C);
    auto BJ = getRandomVector<float>(R * C);

    auto transpose = facebook::sysml::aot::transposer_baseline(
        // Order
        {{"R", 1}, {"C", 1}},
        // Sizes
        {{"R", R}, {"C", C}},
        // Out Strides
        {{"R", 1}, {"C", R}},
        // In Strides
        {{"R", C}, {"C", 1}});

    auto transpose_jit =
        facebook::sysml::aot::transposer_jitter<CT_ISA>(
            {{"R", 1}, {"C", 1}},
            // Sizes
            {{"R", R}, {"C", C}},
            // Out Strides
            {{"R", 1}, {"C", R}},
            // In Strides
            {{"R", C}, {"C", 1}}, 4)
            .get_shared();

    transpose(B.data(), A.data());

    transpose_jit.save_to_file("zi.asm");

    transpose_jit(BJ.data(), A.data());

    std::cout << "MAXABSDIFF: "
              << maxAbsDiff(BJ.data(), BJ.data() + R * C, B.data()) << "\n";

    // for (int r = 0; r < R; ++r)
    // {
    //     for (int c = 0; c < C; ++c)
    //     {
    //         std::cout << A[r * C + c] << "; ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
    // std::cout << "\n";

    // for (int r = 0; r < R; ++r)
    // {
    //     for (int c = 0; c < C; ++c)
    //     {
    //         std::cout << B[r * C + c] << "; ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
    // std::cout << "\n";

    // for (int r = 0; r < R; ++r)
    // {
    //     for (int c = 0; c < C; ++c)
    //     {
    //         std::cout << BJ[r * C + c] << "; ";
    //     }
    //     std::cout << "\n";
    // }

    // for (int r = 0; r < R; ++r)
    // {
    //     for (int c = 0; c < C; ++c)
    //     {
    //         std::cout << BJ[r * C + c] << " :: " << B[r * C + c] << " = "
    //                   << (BJ[r * C + c] - B[r * C + c]) << "; ";
    //     }
    //     std::cout << "\n";
    // }

    // facebook::sysml::aot::transposer_bench<CT_ISA>(
    //     {{"C", 128}, {"R", 128}, {"C", 1}, {"R", 1}}, {{"R", R}, {"C", C}},
    //     {{"R", 1}, {"C", R}}, {{"R", 1}, {"C", C}}, 32, 1, 0);

    // facebook::sysml::aot::transposer_bench<CT_ISA>(
    //     {{"R", 16}, {"R", 1}, {"C", 1}}, {{"R", R}, {"C", C}},
    //     {{"R", C}, {"C", 1}}, {{"R", 1}, {"C", R}}, 16, 1, 0);
}
