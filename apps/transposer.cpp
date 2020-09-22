#include "dabun/transposer.hpp"
#include "dabun/check.hpp"
#include "dabun/random_vector.hpp"
#include "transposer_baseline.hpp"
#include "transposer_bench.hpp"

int main()
{

    using namespace dabun;

    for (int i = 0; i < 10;)
    {

        int R = 11;
        int C = 13;

        auto A  = get_random_vector<float>(R * C);
        auto B  = get_random_vector<float>(R * C);
        auto BJ = get_random_vector<float>(R * C);

        // ArCr=12 AcBr=6 ORDER: ArCr,12 :: AcBr,5 :: AcBr,4 :: ArCr,4 :: AcBr,1
        // :: ArCr,1 :: MU=32

        auto transpose = transposer_baseline(
            // Order
            {{"C", 13}, {"R", 16}, {"C", 9}, {"R", 16}, {"C", 1}, {"R", 1}},
            // Sizes
            {{"R", R}, {"C", C}},
            // Out Strides
            {{"R", C}, {"C", 1}},
            // In Strides
            {{"R", 1}, {"C", R}});

        auto transpose_jit =
            transposer_code_generator<DABUN_ISA>(
                {{"C", 13}, {"R", 16}, {"C", 9}, {"R", 16}, {"C", 1}, {"R", 1}},
                // Sizes
                {{"R", R}, {"C", C}},
                // Out Strides
                {{"R", C}, {"C", 1}},
                // In Strides
                {{"R", 1}, {"C", R}}, 32)
                .get_shared();

        transpose(B.data(), A.data());

        transpose_jit.save_to_file("zi.asm");

        transpose_jit(BJ.data(), A.data());

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(BJ.data(), BJ.data() + R * C, B.data())
                  << "\n";
    }
}
