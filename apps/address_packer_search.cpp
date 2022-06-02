#include <iostream>

#include "dabun/x86/xbyak.hpp"

// template<class F>
auto get_size(bool is_broadcast, int reg_idx)
{
    return [=](int off)
    {
        Xbyak::CodeGenerator cg;
        // f(cg);
        if (is_broadcast)
        {
            cg.vfmadd231ps(cg.zmm0, cg.zmm1,
                           cg.ptr_b[Xbyak::Reg64(reg_idx) + off * 0x4]);
        }
        else
        {
            cg.vfmadd231ps(cg.zmm0, cg.zmm1,
                           cg.ptr[Xbyak::Reg64(reg_idx) + off * 0x40]);
        }
        return cg.getSize();
    };
}

auto get_size2(bool is_broadcast, int reg_idx)
{
    return [=](int off)
    {
        Xbyak::CodeGenerator cg;
        // f(cg);
        if (is_broadcast)
        {
            cg.vmovups(cg.zmm0, cg.ptr_b[Xbyak::Reg64(reg_idx) + off * 0x4]);
        }
        else
        {
            cg.vmovups(cg.zmm0, cg.ptr[Xbyak::Reg64(reg_idx) + off * 0x40]);
        }
        return cg.getSize();
    };
}

template <class F>
int binary_search(F const& f, int begin, int end, int s)
{
    if (begin == end)
    {
        return begin;
    }

    int mid = begin + (end - begin) / 2;

    if (f(mid) == s)
    {
        return binary_search(f, mid + 1, end, s);
    }
    else
    {
        return binary_search(f, begin, mid, s);
    }
}

template <class F>
int do_search(F const& f)
{
    std::cout << "F(0) = " << f(0) << "; F(1) = " << f(1) << "\n";
    int  s   = f(1);
    auto ret = binary_search(f, 1, 0xFFFFFF, s);
    return ret;
}

int main()
{
    auto fn = get_size2(true, 0);

    std::cout << "line ";
    std::cout << fn(0) << ' ';
    std::cout << fn(10) << ' ';
    std::cout << do_search(fn) << ' ';
    std::cout << std::endl;
}
