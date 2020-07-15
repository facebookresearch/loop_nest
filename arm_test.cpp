#include "code_generator.h"
#include "multi_vreg.h"
#include "math.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

class test : public facebook::sysml::aot::code_generator<void(float const*,
                                                              float*, float*)>
{
public:
    test()
    {
        using namespace Xbyak;

        // ld4((v0 - v3), ptr(x0));

        ld4({v0.s4 - v3.s4}, ptr(x0));
        ldp(q24, q25, post_ptr(x1, 64));
        ldp(q8, q9, ptr(x2));

        fmla(v8.s4, v24.s4, v0.s[0]);
        fmla(v9.s4, v25.s4, v0.s[0]);

        stp(q8, q9, ptr(x2));

        ret();
    }
};

class test2
    : public facebook::sysml::aot::code_generator<int(float const*, float*)>
{
public:
    test2()
    {
        // using namespace Xbyak;

        // eor(x4, x4, x4);

        // ld4((v0.b - v3.b), ptr(x0));

        // ld4({v0.s4 - v3.s4}, ptr(x0));
        // ldp(q24, q25, post_ptr(x1, 64));
        // ldp(q8, q9, ptr(x2));

        // fmla(v8.s4, v24.s4, v0.s[0]);
        // fmla(v9.s4, v25.s4, v0.s[0]);

        // stp(q8, q9, ptr(x2));

        // ldp(q0, q1, ptr(x0));
        // facebook::sysml::aot::multi_vreg<VReg> mv(2, 0);
        // //mv.full_reduce(*this, 3);

        // // mov(v0.s[0], w4);
        // // dup(v0.s4, v0.s[0]);
        // //fmov(v0.s4, 0.0);

        // st1(v0.s4, ptr(x1));

        // // st1(v1.s4, ptr(x1));


        // sub(sp, sp, 256);
        // mov(w2, 16);
        // str(w2, ptr(sp, 32));
        // ldr(w0, ptr(sp, 32));
        // add(sp, sp, 256);

        // eor(x0, x0, x0);
        // //eors(x0, x0, x0);
        // cmp(x0, 0);

        ld1(v1.s4, ptr(x1));

        ld1(v2.s4, ptr(x0));
        ld1(v3.s[0], ptr(x0));
        ld1(v3.s[1], ptr(x0));
        ld1(v3.s[2], ptr(x0));
        ld1(v3.s[3], ptr(x0));
        //ld1(v3.s[0], ptr(x1));

        fmla(v1.s4, v3.s4, v2.s4);

        st1(v1.s4, ptr(x1));

        ret();
    }
};

int main()
{
    float A[16] = {2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
                   2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
    float B[16] = {2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
                   2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
    float C[16] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // auto fn = test().get_shared();
    // fn.save_to_file("zi.asm");
    // fn(A, B, C);

    auto fn = test2().get_shared();
    fn.save_to_file("zi.asm");
    std::cout << "FN: " << fn(A, C) << "\n";

    for (int i = 0; i < 8; ++i)
    {
        std::cout << "C[" << i << "] = " << C[i] << "\n";
    }
}
