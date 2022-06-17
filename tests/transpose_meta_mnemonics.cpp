
#include <catch2/catch.hpp>

#include "dabun/code_generator/code_generator.hpp"
#include "dabun/random_vector.hpp"
#include "sysml/miltuple.hpp"
#include "sysml/random.hpp"

#include <tuple>
#include <utility>

using namespace dabun;

template <class T>
void print_array2d(T const* a, int rows, int cols, int row_stride = -1,
                   int col_stride = 1)
{
    if (row_stride == -1)
        row_stride = cols;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            if (c != 0)
                std::cout << ' ';
            auto val = (int)a[r * row_stride + c * col_stride];
            if (val < 10)
                std::cout << "00";
            else if (val < 100)
                std::cout << "0";
            std::cout << val;
        }

        std::cout << "\n";
    }
}

class tester : public dabun::code_generator<void(double*, double*)>
{
public:
    tester()
    {
        ld1(v0.d, ptr(x0));
        ld1(v1.d, ptr(x1));
        // st1(v1.d, ptr(x0));
        // st1(v0.d, ptr(x1));
        trn1(v2.d2, v0.d2, v1.d2);
        trn2(v3.d2, v0.d2, v1.d2);
        st1(v2.d2, ptr(x0));
        st1(v3.d2, ptr(x1));

        ret();
    }
};

class transposer_2x2_fp64 : public dabun::code_generator<void(double*, double*)>
{
public:
    transposer_2x2_fp64()
    {
        ld1(v0.d, ptr(x0));
        ld1(v1.d, ptr(x1));
        // st1(v1.d, ptr(x0));
        // st1(v0.d, ptr(x1));
        zip1(v2.d2, v0.d2, v1.d2);
        zip2(v3.d2, v0.d2, v1.d2);
        st1(v2.d2, ptr(x0));
        st1(v3.d2, ptr(x1));

        ret();
    }
};

class transposer_4x4_32
    : public dabun::code_generator<void(float*, float*, float*, float*)>
{
public:
    transposer_4x4_32()
    {
        ld1(v0.s, ptr(x0));
        ld1(v1.s, ptr(x1));
        ld1(v2.s, ptr(x2));
        ld1(v3.s, ptr(x3));
        // st1(v1.d, ptr(x0));
        // st1(v0.d, ptr(x1));
        trn1(v4.s4, v0.s4, v1.s4);
        trn2(v5.s4, v0.s4, v1.s4);

        trn1(v6.s4, v2.s4, v3.s4);
        trn2(v7.s4, v2.s4, v3.s4);

        // 5->0, 6->1, 7->2

        zip1(v0.d2, v4.d2, v6.d2);
        zip2(v2.d2, v4.d2, v6.d2);

        zip1(v1.d2, v5.d2, v7.d2);
        zip2(v3.d2, v5.d2, v7.d2);

        // zip1(v2.s4, v0.s4, v1.s4);
        // zip2(v3.s4, v0.s4, v1.s4);
        // st1(v2.d2, ptr(x0));
        // st1(v3.d2, ptr(x1));

        st1(v0.s, ptr(x0));
        st1(v1.s, ptr(x1));
        st1(v2.s, ptr(x2));
        st1(v3.s, ptr(x3));

        ret();
    }
};

class transposer
    //    : public dabun::code_generator<void(float*, float*, float*, float*)>
    : public dabun::code_generator<void(void*)>
{
private:
    template <class F>
    void slide_transpose_2x2(VReg const& in_r0, VReg const& in_r1_out_r0,
                             VReg const& out_r1, F&& f)
    {
        zip2(f(out_r1), f(in_r0), f(in_r1_out_r0));
        zip1(f(in_r1_out_r0), f(in_r0), f(in_r1_out_r0));
    }

    template <class F>
    void transpose_2x2_keep_first(VReg const& r0, VReg const& r1,
                                  VReg const& tmp, F&& f)
    {
        trn2(f(tmp), f(r0), f(r1));
        trn1(f(r0), f(r0), f(r1));
    }

    template <class F>
    void transpose_2x2_keep_second(VReg const& r0, VReg const& r1,
                                   VReg const& tmp, F&& f)
    {
        trn1(f(tmp), f(r0), f(r1));
        trn2(f(r1), f(r0), f(r1));
    }

    template <unsigned ElementSize>
    void transpose_4x4_inplace(sysml::miltuple<VReg const&, 4> const& r,
                               sysml::miltuple<VReg const&, 2> const& tmp,
                               vreg_view<ElementSize>                 vview)
    {
        auto const& [r0, r1, r2, r3] = r;
        auto const& [t0, t1]         = tmp;

        transpose_2x2_keep_first(r0, r1, t0, vview);
        transpose_2x2_keep_second(r2, r3, t1, vview);

        transpose_2x2_keep_first(r0, t1, r2, vreg_view<ElementSize * 2>());
        transpose_2x2_keep_second(t0, r3, r1, vreg_view<ElementSize * 2>());

        // trn2(onef(t0), onef(r0), onef(r1));
        // trn1(onef(r0), onef(r0), onef(r1));

        // trn1(onef(t1), onef(r2), onef(r3));
        // trn2(onef(r3), onef(r2), onef(r3));

        // trn2(twof(r2), twof(r0), twof(t1));
        // trn1(twof(r0), twof(r0), twof(t1));

        // trn1(twof(r1), twof(t0), twof(r3));
        // trn2(twof(r3), twof(t0), twof(r3));
    }

    template <unsigned ElementSize>
    void transpose_8x8_inplace(sysml::miltuple<VReg const&, 8> const& r,
                               sysml::miltuple<VReg const&, 2> const& tmp,
                               vreg_view<ElementSize>                 vview)
    {
        auto const& [r0, r1, r2, r3, r4, r5, r6, r7] = r;
        auto const& [t0, t1]                         = tmp;

        vreg_view<ElementSize * 2> dview;
        vreg_view<ElementSize * 4> qview;

        transpose_4x4_inplace(std::tie(r4, r5, r6, r7), std::tie(t0, t1),
                              vview);

        transpose_2x2_keep_first(r0, r1, t0, vview);
        transpose_2x2_keep_second(r2, r3, t1, vview);

        transpose_2x2_keep_second(r0, t1, r1, dview);
        transpose_2x2_keep_second(r1, r4, r0, qview);
        transpose_2x2_keep_second(t1, r6, r2, qview);

        transpose_2x2_keep_first(t0, r3, t1, dview);
        transpose_2x2_keep_second(t0, r5, r1, qview);
        transpose_2x2_keep_second(t1, r7, r3, qview);

        // transpose_4x4_inplace(std::tie(r0, r1, r2, r3), std::tie(t0, t1),
        //                       vview);

        // transpose_2x2_keep_first(r0, t0, r4, vreg_view<ElementSize * 4>());
        // transpose_2x2_keep_first(r1, t1, r5, vreg_view<ElementSize * 4>());
        // transpose_2x2_keep_first(r2, t2, r6, vreg_view<ElementSize * 4>());
        // transpose_2x2_keep_first(r3, t3, r7, vreg_view<ElementSize * 4>());
    }

    template <unsigned ElementSize>
    void transpose_16x16_inplace(sysml::miltuple<VReg const&, 16> const& r,
                                 sysml::miltuple<VReg const&, 2> const&  tmp,
                                 vreg_view<ElementSize>                  vview)
    {
        auto const& [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13,
                     r14, r15] = r;
        // auto const& [t0, t1]                 = tmp;

        transpose_4x4_inplace(std::tie(r0, r1, r2, r3), tmp, vview);
        transpose_4x4_inplace(std::tie(r4, r5, r6, r7), tmp, vview);
        transpose_4x4_inplace(std::tie(r8, r9, r10, r11), tmp, vview);
        transpose_4x4_inplace(std::tie(r12, r13, r14, r15), tmp, vview);

        vreg_view<ElementSize * 4> qview;

        transpose_4x4_inplace(std::tie(r0, r4, r8, r12), tmp, qview);
        transpose_4x4_inplace(std::tie(r1, r5, r9, r13), tmp, qview);
        transpose_4x4_inplace(std::tie(r2, r6, r10, r14), tmp, qview);
        transpose_4x4_inplace(std::tie(r3, r7, r11, r15), tmp, qview);

        // transpose_2x2_keep_first(r0, t0, r4, vreg_view<ElementSize * 4>());
        // transpose_2x2_keep_first(r1, t1, r5, vreg_view<ElementSize * 4>());
        // transpose_2x2_keep_first(r2, t2, r6, vreg_view<ElementSize * 4>());
        // transpose_2x2_keep_first(r3, t3, r7, vreg_view<ElementSize * 4>());
    }

    template <unsigned ElementSize>
    void transpose_4x4_onto(sysml::miltuple<VReg const&, 4> const& src,
                            sysml::miltuple<VReg const&, 4> const& dst,
                            vreg_view<ElementSize>                 vview)

    {
        auto const& [src0, src1, src2, src3] = src;
        auto const& [dst0, dst1, dst2, dst3] = dst;

        transpose_2x2_keep_first(src0, src1, dst1, vview);
        transpose_2x2_keep_second(src2, src3, dst2, vview);

        transpose_2x2_keep_second(src0, dst2, dst0,
                                  vreg_view<ElementSize * 2>());
        transpose_2x2_keep_first(dst1, src3, dst3,
                                 vreg_view<ElementSize * 2>());

        // trn2(onef(dst1), onef(src0), onef(src1));
        // trn1(onef(src0), onef(src0), onef(src1));

        // trn1(onef(dst2), onef(src2), onef(src3));
        // trn2(onef(src3), onef(src2), onef(src3));

        // trn1(twof(dst0), twof(src0), twof(dst2));
        // trn2(twof(dst2), twof(src0), twof(dst2));

        // trn2(twof(dst3), twof(dst1), twof(src3));
        // trn1(twof(dst1), twof(dst1), twof(src3));
    }

public:
    // void transpose_4x4_32(VReg const& v0, VReg const& v1, VReg const& v2,
    //                       VReg const& v3, VReg const& tmp0, VReg const& tmp1)
    // {
    //     transpose_4x4_inplace(
    //         std::tie(v0, v1, v2, v3), std::tie(tmp0, tmp1),
    //         [](VReg const& r) { return r.s4; },
    //         [](VReg const& r) { return r.d2; });
    // }

    transposer(int s)
    {
        ld1(v0.h, ptr(x0));
        add(x0, x0, 16);
        ld1(v1.h, ptr(x0));
        add(x0, x0, 16);
        ld1(v2.h, ptr(x0));
        add(x0, x0, 16);
        ld1(v3.h, ptr(x0));
        add(x0, x0, 16);
        ld1(v4.h, ptr(x0));
        add(x0, x0, 16);
        ld1(v5.h, ptr(x0));
        add(x0, x0, 16);
        ld1(v6.h, ptr(x0));
        add(x0, x0, 16);
        ld1(v7.h, ptr(x0));

        if (s == 16)
        {
            add(x0, x0, 16);
            ld1(v16.h, ptr(x0));
            add(x0, x0, 16);
            ld1(v17.h, ptr(x0));
            add(x0, x0, 16);
            ld1(v18.h, ptr(x0));
            add(x0, x0, 16);
            ld1(v19.h, ptr(x0));
            add(x0, x0, 16);
            ld1(v20.h, ptr(x0));
            add(x0, x0, 16);
            ld1(v21.h, ptr(x0));
            add(x0, x0, 16);
            ld1(v22.h, ptr(x0));
            add(x0, x0, 16);
            ld1(v23.h, ptr(x0));
        }

        // transpose_4x4_32(v0, v1, v2, v3, v4, v5);

        if (s == 16)
        {

            transpose_16x16_inplace(std::tie(v0, v1, v2, v3, v4, v5, v6, v7,
                                             v16, v17, v18, v19, v20, v21, v22,
                                             v23),
                                    std::tie(v24, v25), vreg_view<1>());
        }
        else
        {

            transpose_8x8_inplace(std::tie(v0, v1, v2, v3, v4, v5, v6, v7),
                                  std::tie(v24, v25), vreg_view<2>());
        }
        // trn1(v0.h8, v0.h8, v1.h8);
        // // trn2(v5.s4, v0.s4, v1.s4);

        // trn1(v6.s4, v2.s4, v3.s4);
        // trn2(v7.s4, v2.s4, v3.s4);
        if (s == 16)
        {

            st1(v23.h, ptr(x0));
            sub(x0, x0, 16);
            st1(v22.h, ptr(x0));
            sub(x0, x0, 16);
            st1(v21.h, ptr(x0));
            sub(x0, x0, 16);
            st1(v20.h, ptr(x0));
            sub(x0, x0, 16);
            st1(v19.h, ptr(x0));
            sub(x0, x0, 16);
            st1(v18.h, ptr(x0));
            sub(x0, x0, 16);
            st1(v17.h, ptr(x0));
            sub(x0, x0, 16);
            st1(v16.h, ptr(x0));
            sub(x0, x0, 16);
        }

        st1(v7.h, ptr(x0));
        sub(x0, x0, 16);
        st1(v6.h, ptr(x0));
        sub(x0, x0, 16);
        st1(v5.h, ptr(x0));
        sub(x0, x0, 16);
        st1(v4.h, ptr(x0));
        sub(x0, x0, 16);
        st1(v3.h, ptr(x0));
        sub(x0, x0, 16);
        st1(v2.h, ptr(x0));
        sub(x0, x0, 16);
        st1(v1.h, ptr(x0));
        sub(x0, x0, 16);
        st1(v0.h, ptr(x0));

        ret();
    }
};

TEST_CASE("ok?", "[wtfgh]")
{
    if (1)
    {
        double r0[] = {1.0, 2.0};
        double r1[] = {3.0, 4.0};

        auto fn = tester().get_unique();
        fn(r0, r1);

        std::cout << "r0: " << r0[0] << ", " << r0[1] << "\n";
        std::cout << "r1: " << r1[0] << ", " << r1[1] << "\n";
    }

    // if (1)
    // {
    //     float r0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    //     float r1[] = {5.0f, 6.0f, 7.0f, 8.0f};
    //     float r2[] = {9.0f, 10.0f, 11.0f, 12.0f};
    //     float r3[] = {13.0f, 14.0f, 15.0f, 16.0f};

    //     auto fn = transposer().get_unique();

    //     fn(r0, r1, r2, r3);

    //     std::cout << "r0: " << r0[0] << ", " << r0[1] << ", " << r0[2] << ",
    //     "
    //               << r0[3] << "\n";

    //     std::cout << "r1: " << r1[0] << ", " << r1[1] << ", " << r1[2] << ",
    //     "
    //               << r1[3] << "\n";

    //     std::cout << "r2: " << r2[0] << ", " << r2[1] << ", " << r2[2] << ",
    //     "
    //               << r2[3] << "\n";

    //     std::cout << "r3: " << r3[0] << ", " << r3[1] << ", " << r3[2] << ",
    //     "
    //               << r3[3] << "\n";
    // }

    if (1)
    {
        auto m = dabun::get_random_vector<std::uint16_t>(64);

        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 8; ++c)
            {
                m[r * 8 + c] = c + r * 10;
            }

        print_array2d(m.data(), 8, 8);

        auto fn = transposer(8).get_unique();

        fn(m.data());

        std::cout << "\n\n";

        print_array2d(m.data(), 8, 8);

        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 8; ++c)
            {
                REQUIRE(m[r * 8 + c] == r + c * 10);
            }

        std::cout << "\n\n";
        std::cout << "\n\n";

        // dabun::tests::baseline::for_all_elements_of_two_array2d(
    }

    {
        auto m = dabun::get_zero_vector<std::uint8_t>(1256);

        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c)
            {
                m[r * 16 + c] = c + r * 16;
            }

        print_array2d(m.data(), 16, 16);

        auto fn = transposer(16).get_unique();

        fn(m.data());

        std::cout << "\n\n";

        print_array2d(m.data(), 16, 16);

        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c)
            {
                REQUIRE(m[r * 16 + c] == r + c * 16);
            }

        // dabun::tests::baseline::for_all_elements_of_two_array2d(
    }
}
