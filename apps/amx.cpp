#include "dabun/code_generator.hpp"
#include "dabun/isa.hpp"
#include "dabun/qvec4.hpp"

#include "dabun/random_vector.hpp"

#include <iostream>
#include <random>

struct tile_config_t
{
    std::uint8_t  palette          = 1;   // byte 0
    std::uint8_t  start_row        = 0;   // byte 1
    std::uint8_t  reserved1[14]    = {0}; // bytes 2-15
    std::uint16_t bytes_per_row[8] = {64, 64, 64, 64,
                                      64, 64, 64, 64}; // bytes 16-31
    std::uint8_t  reserved2[16]    = {0};              // bytes 32-47
    std::uint8_t  num_rows[8]      = {
        16, 16, 16, 16, 16, 16, 16, 16,
    };                         // bytes 48-55
    std::uint8_t reserved3[8] = {0}; // bytes 56-63
};

static_assert(sizeof(tile_config_t) == 64);

inline void print_tile_config(tile_config_t const& tc)
{
    std::cout << "Palette: " << static_cast<int>(tc.palette) << "\n";
    std::cout << "Start Row: " << static_cast<int>(tc.start_row) << "\n";
    for (int i = 0; i < 8; ++i)
    {
        std::cout << "  Tile " << i
                  << " :: " << static_cast<int>(tc.num_rows[i]) << " x "
                  << tc.bytes_per_row[i] << " bytes\n";
    }
}

class test : public dabun::code_generator<void(void*, void*, void*, void*, void*)>
{
public:
    test()
    {
        ldtilecfg(ptr[rdi]);
        sttilecfg(ptr[rdi]);

        mov(rax, 64);

        tilezero(tmm0);
        tileloadd(tmm1, ptr[rsi + rax * 1]);
        tileloadd(tmm2, ptr[rdx + rax * 1]);

        tdpbuud(tmm0, tmm1, tmm2);

        // ldtilecfg(ptr[r8]);

        tilestored(ptr[rcx + rax * 1], tmm0);

        tilerelease();
        ret();
    }
};

int main()
{
    dabun::uint8x4_t l(1, 2, 3, 4);
    dabun::uint8x4_t r(1, 2, 3, 4);

    std::cout << dot(l, r) << "\n";
    std::cout << vnni_fma(l, r, 0) << "\n";

    for (int i = 0; i < 4; ++i)
    {
        std::cout << "l[" << i << "] = " << l.extended(i) << "\n";
        std::cout << "r[" << i << "] = " << r.extended(i) << "\n";
    }

    std::cout << l << "\n";

    auto          t = test().get_shared();

    auto A  = dabun::get_random_qvector<dabun::uint8x4_t>(16 * 16);
    auto B  = dabun::get_random_qvector<dabun::uint8x4_t>(16 * 16);
    auto CJ = dabun::get_zero_vector<std::int32_t>(16 * 16);
    auto CN = dabun::get_zero_vector<std::int32_t>(16 * 16);

    for (int m = 0; m < 16; ++m)
    {
        for (int n = 0; n < 16; ++n)
        {
            for (int k = 0; k < 16; ++k)
            {
                CN[m * 16 + n] += dot(A[m * 16 + k], B[k * 16 + n]);
            }
        }
    }

    tile_config_t tc;
    tile_config_t tc2;
    tc2.num_rows[0] = 4;
    t(&tc, A.data(), B.data(), CJ.data(), &tc2);
    print_tile_config(tc);

    for (int i = 0; i < 16 * 16; ++i)
    {
        std::cout << CN[i] << " -- " << CJ[i] << "\n";
    }
}
