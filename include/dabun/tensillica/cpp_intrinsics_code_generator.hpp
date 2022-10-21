// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "dabun/tensillica/multi_vmm.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>

namespace dabun
{
namespace tensillica
{
// namespace Xbyak
// {

inline std::uint64_t next_dl_instance_id() noexcept
{
    static std::atomic<std::uint64_t> dl_counter{0};
    return dl_counter++;
}

inline std::pair<void*, void*> compile_to_dl(std::string const& fn_name,
                                             std::string const& fn_text)
{
    std::string source_fname = std::string("/tmp/") + fn_name + ".cpp";

#ifdef __APPLE__
    std::string dl_fname = std::string("/tmp/") + fn_name + ".dylib";
    std::string compile_command =
        std::string("g++ -std=c++20 -dynamiclib -O3 -DNDEBUG -fPIC -o ") +
        dl_fname + " " + source_fname;
#else
    std::string so_name  = fn_name + ".so";
    std::string dl_fname = std::string("/tmp/") + so_name;
    std::string compile_command =
        std::string("g++ -std=c++20 -shared -Wl,-soname,") + so_name +
        " -O3 -DNDEBUG -fPIC -o " + dl_fname + " " + source_fname;
#endif

    {
        std::ofstream ofs(source_fname,
                          std::ios_base::out | std::ios_base::trunc);
        ofs << fn_text;
    }

    std::system(compile_command.c_str());
    std::remove(source_fname.c_str());

    void* dlh = ::dlopen(dl_fname.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (dlh)
    {
        void* dls = ::dlsym(dlh, fn_name.c_str());
        if (dls)
        {
            return {dlh, dls};
        }
    }

    std::remove(dl_fname.c_str());

    return {nullptr, nullptr};
}

template <class To, class From>
typename std::enable_if_t<
    std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>, To>
// constexpr support needs compiler magic
dl_func_arg_cast(const From& src) noexcept
{
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation additionally requires destination type "
                  "to be trivially constructible");

    To dst;

    if constexpr (sizeof(To) > sizeof(From))
    {
        std::memset(&dst, 0, sizeof(From));
        std::memcpy(&dst, &src, sizeof(From));
    }
    else
    {
        std::memcpy(&dst, &src, sizeof(To));
    }

    return dst;
}

template <class T>
struct is_supported_dl_fn_type
    : std::integral_constant<bool, std::is_trivially_copyable_v<T> &&
                                       (sizeof(T) == 8 || sizeof(T) == 16 ||
                                        sizeof(T) == 32 || sizeof(T) == 64)>
{
};

template <class T>
constexpr bool is_supported_dl_fn_type_v = is_supported_dl_fn_type<T>::value;

template <class, class T>
struct second_type
{
    using type = T;
};

template <class F, class S>
using second_type_t = typename second_type<F, S>::type;

template <typename Signature>
class shared_dl_fn;

template <typename ReturnType, typename... Args>
class shared_dl_fn<ReturnType(Args...)>
{
private:
    // Do static asserts

private:
    std::shared_ptr<void> executable_buffer_;

public:
    template <typename Deleter>
    shared_dl_fn(void* buffer, Deleter deleter)
        : executable_buffer_(buffer, deleter)
    {
    }

    shared_dl_fn() noexcept {}

    shared_dl_fn(shared_dl_fn const&) = default;
    shared_dl_fn& operator=(shared_dl_fn const&) = default;
    shared_dl_fn(shared_dl_fn&&)                 = default;
    shared_dl_fn& operator=(shared_dl_fn&&) = default;

    ReturnType operator()(Args... args) const
    {
        auto callable = reinterpret_cast<std::conditional_t<
            std::is_same_v<ReturnType, void>, void, std::int64_t> (*)(
            second_type_t<Args, std::int64_t>...)>(executable_buffer_.get());

        if constexpr (std::is_same_v<ReturnType, void>)
        {
            callable(
                dl_func_arg_cast<second_type_t<Args, std::int64_t>>(args)...);
        }
        else
        {
            auto ret = callable(
                dl_func_arg_cast<second_type_t<Args, std::int64_t>>(args)...);
            return dl_func_arg_cast<ReturnType>(ret);
        }
    }

    explicit operator bool() const noexcept { return !!executable_buffer_; }
};

template <unsigned LaneSize>
struct GenericVRegLane
{
    int vidx;
    int lidx;
};

template <unsigned LaneSize, unsigned NumLanes>
struct GenericVReg
{
    int idx_;

    int idx() const { return idx_; }

    int getIdx() const { return idx_; }

    explicit GenericVReg(int i)
        : idx_(i)
    {
    }
    GenericVReg()                   = default;
    ~GenericVReg()                  = default;
    GenericVReg(GenericVReg const&) = default;
    GenericVReg(GenericVReg&&)      = default;
    GenericVReg& operator=(GenericVReg const&) = default;
    GenericVReg& operator=(GenericVReg&&) = default;

    GenericVRegLane<LaneSize> operator[](int l) const { return {idx_, l}; }
};

using SReg = GenericVReg<4, 1>;
using DReg = GenericVReg<4, 2>;
using QReg = GenericVReg<4, 4>;

struct Vmm
{
    int idx_;

    int idx() const { return idx_; }
    int getIdx() const { return idx_; }

    Vmm() {}

    Vmm(Vmm const&) = default;
    Vmm(Vmm&&)      = default;

    Vmm& operator=(Vmm const&) = default;
    Vmm& operator=(Vmm&&) = default;

    explicit Vmm(int i)
        : idx_(i)
        , s{i}
        , s4{i}
        , s2{i}
        , s1{i}
        , d{i}
        , d2{i}
        , d1{i}
    {
    }

    GenericVReg<4, 4> s;
    GenericVReg<4, 4> s4;
    GenericVReg<4, 2> s2;
    GenericVReg<4, 1> s1;

    GenericVReg<8, 2> d;
    GenericVReg<8, 2> d2;
    GenericVReg<8, 1> d1;
};

struct Reg64
{
    int idx_;

    int idx() const { return idx_; }
    int getIdx() const { return idx_; }

    Reg64()  = default;
    ~Reg64() = default;
    explicit Reg64(int i)
        : idx_(i)
    {
    }
    Reg64(Reg64 const&) = default;
    Reg64(Reg64&&)      = default;
    Reg64& operator=(Reg64 const&) = default;
    Reg64& operator=(Reg64&&) = default;
};

struct Reg32
{
    int idx_;

    int idx() const { return idx_; }
    int getIdx() const { return idx_; }

    Reg32()  = default;
    ~Reg32() = default;
    explicit Reg32(int i)
        : idx_(i)
    {
    }
    Reg32(Reg32 const&) = default;
    Reg32(Reg32&&)      = default;
    Reg32& operator=(Reg32 const&) = default;
    Reg32& operator=(Reg32&&) = default;
};

struct ptr_impl
{
    Reg64 reg;
};

struct pre_ptr_impl
{
    Reg64        reg;
    std::int64_t pre;
};

struct post_ptr_impl
{
    Reg64        reg;
    std::int64_t post;
};

inline ptr_impl ptr(Reg64 const& x) { return {x}; }

inline pre_ptr_impl pre_ptr(Reg64 const& x, std::int64_t pre)
{
    return {x, pre};
}

inline post_ptr_impl post_ptr(Reg64 const& x, std::int64_t post)
{
    return {x, post};
}

template <class>
class cpp_intrinsics_code_generator;

template <class ReturnType, class... Args>
class cpp_intrinsics_code_generator<ReturnType(Args...)>
{
private:
    static constexpr unsigned NumArgs = sizeof...(Args);

private:
    std::ostringstream oss_;

private:
    static void issue_prologue(std::ostringstream& oss,
                               std::string         name = "noname")
    {
        oss << "#include <cstdint>\n#include <cassert>\n#include "
               "<algorithm>\n#include "
               "<vector>\n#include "
               "<bit>\n#include <cstring>\n";
        oss << "#include <type_traits>\n";
        oss << "template <class To, class From>\n"
            << "typename std::enable_if_t<sizeof(To) == sizeof(From) &&\n"
            << "                              "
            << "std::is_trivially_copyable_v<From> &&\n"
            << "                              "
            << "std::is_trivially_copyable_v<To>,\n"
            << "                          To>\n"
            << "bit_cast(const From& src) noexcept \n"
            << "{\n"
            << "    static_assert(std::is_trivially_constructible_v<To>,\n"
            << "                  \"This implementation additionally requires "
            << "destination type \"\n"
            << "                  \"to be trivially constructible\");\n"
            << "    To dst;\n"
            << "    std::memcpy(&dst, &src, sizeof(To));\n"
            << "    return dst;\n"
            << "}\n";

        oss << "#define as(t, w) bit_cast<t>(w)\n";
        oss << "extern \"C\" ";
        oss << (std::is_same_v<ReturnType, void> ? "void" : "std::int64_t");
        oss << " " << name << "(";
        for (unsigned i = 0; i < NumArgs; ++i)
        {
            if (i)
            {
                oss << ", ";
            }
            oss << "std::int64_t "
                << " function_arg" << i;
        }
        oss << ")\n";

        oss << "{\n";
        // Scalar Regs
        for (unsigned i = 0; i < 32; ++i)
        {
            oss << "  std::int64_t x" << i << ";\n";
        }

        oss << "\n";

        // Vector Regs
        for (unsigned i = 0; i < 32; ++i)
        {
            oss << "  float vmm" << i << "[4];\n";
        }

        oss << "\n";

        for (unsigned i = 0; i < NumArgs; ++i)
        {
            oss << "  x" << i << " = function_arg" << i << ";\n";
        }

        oss << "\n";

        oss << "  std::int64_t cmp_lhs;\n";
        oss << "  std::int64_t cmp_rhs;\n";

        oss << "\n";
    }

public:
    void custom_string(std::string const& s) { oss_ << s << "\n"; }

private:
    void issue_epilogue() { oss_ << "}\n"; }

public:
    void restore_stack() {}

    void ret() {}

    void prepare_stack()
    {
        oss_ << "  std::vector<std::uint64_t> the_stack(2048);\n  "
                "int stack_idx = 0;\n\n";
    }

    void mov(Reg64 dst, Reg64 src)
    {
        oss_ << "  x" << dst.idx() << " = x" << src.idx() << ";\n";
    }

    void and_(Reg64 dst, Reg64 src, int v)
    {
        oss_ << "  x" << dst.idx() << " = (x" << src.idx() << " & " << v
             << ");\n";
    }

    void meta_push(Reg64 r)
    {
        oss_ << "  the_stack[stack_idx++] = as(std::uint64_t, x" << r.idx()
             << ");\n";
    }

    void meta_pop(Reg64 r)
    {
        oss_ << "  x" << r.idx()
             << " = as(std::int64_t, the_stack[--stack_idx]);\n";
    }

    void meta_push(std::vector<Reg64> const& regs)
    {
        for (auto r : regs)
        {
            meta_push(r);
        }
    }

    void meta_pop(std::vector<Reg64> regs)
    {
        std::reverse(regs.begin(), regs.end());
        for (auto r : regs)
        {
            meta_pop(r);
        }
    }

    void newline() { oss_ << "\n"; }

    void custom(std::string const& s) { oss_ << "  " << s << " // CUSTOM\n"; }

    shared_dl_fn<ReturnType(Args...)> get_shared() &&
    {
        issue_epilogue();

        std::string fn_name = "noname_" + std::to_string(next_dl_instance_id());

        std::ostringstream oss;
        issue_prologue(oss, fn_name);

        oss << oss_.str();
        std::string fn_text = oss.str();

        std::cout << fn_text;

        auto handle = compile_to_dl(fn_name, fn_text);

        if (handle.first && handle.second)
        {
            return shared_dl_fn<ReturnType(Args...)>(
                handle.second, [dlh = handle.first](void*) { ::dlclose(dlh); });
        }

        return {};
    }

    cpp_intrinsics_code_generator() {}

public:
    Reg64 x0{0}, x1{1}, x2{2}, x3{3};
    Reg64 x4{4}, x5{5}, x6{6}, x7{7};
    Reg64 x8{8}, x9{9}, x10{10}, x11{11};
    Reg64 x12{12}, x13{13}, x14{14}, x15{15};
    Reg64 x16{16}, x17{17}, x18{18}, x19{19};
    Reg64 x20{20}, x21{21}, x22{22}, x23{23};
    Reg64 x24{24}, x25{25}, x26{26}, x27{27};
    Reg64 x28{28}, x29{29}, x30{30}, x31{31};

    Reg32 w0{0}, w1{1}, w2{2}, w3{3};
    Reg32 w4{4}, w5{5}, w6{6}, w7{7};
    Reg32 w8{8}, w9{9}, w10{10}, w11{11};
    Reg32 w12{12}, w13{13}, w14{14}, w15{15};
    Reg32 w16{16}, w17{17}, w18{18}, w19{19};
    Reg32 w20{20}, w21{21}, w22{22}, w23{23};
    Reg32 w24{24}, w25{25}, w26{26}, w27{27};
    Reg32 w28{28}, w29{29}, w30{30}, w31{31};

    Vmm vmm0{0}, vmm1{1}, vmm2{2}, vmm3{3};
    Vmm vmm4{4}, vmm5{5}, vmm6{6}, vmm7{7};
    Vmm vmm8{8}, vmm9{9}, vmm10{10}, vmm11{11};
    Vmm vmm12{12}, vmm13{13}, vmm14{14}, vmm15{15};
    Vmm vmm16{16}, vmm17{17}, vmm18{18}, vmm19{19};
    Vmm vmm20{20}, vmm21{21}, vmm22{22}, vmm23{23};
    Vmm vmm24{24}, vmm25{25}, vmm26{26}, vmm27{27};
    Vmm vmm28{28}, vmm29{29}, vmm30{30}, vmm31{31};

    Vmm v0{0}, v1{1}, v2{2}, v3{3};
    Vmm v4{4}, v5{5}, v6{6}, v7{7};
    Vmm v8{8}, v9{9}, v10{10}, v11{11};
    Vmm v12{12}, v13{13}, v14{14}, v15{15};
    Vmm v16{16}, v17{17}, v18{18}, v19{19};
    Vmm v20{20}, v21{21}, v22{22}, v23{23};
    Vmm v24{24}, v25{25}, v26{26}, v27{27};
    Vmm v28{28}, v29{29}, v30{30}, v31{31};

private:
    template <unsigned NumLanes>
    std::string to_string(GenericVRegLane<NumLanes> lane)
    {
        std::ostringstream oss;
        oss << "vmm" << lane.vidx << "[" << lane.lidx << "]";
        return oss.str();
    }

public:
    void eor(Reg64 dst, Reg64 src1, Reg64 src2)
    {
        oss_ << "  x" << dst.idx() << " = x" << src1.idx() << " ^ x"
             << src2.idx() << ";\n";
    }

    void eor(Reg32 dst, Reg32 src1, Reg32 src2)
    {
        oss_ << "  x" << dst.idx() << " = x" << src1.idx() << " ^ x"
             << src2.idx() << ";\n";
    }

    void ins(GenericVRegLane<4> dst, Reg32 src)
    {
        oss_ << "  " << to_string(dst) << " = "
             << "as(float, static_cast<std::uint32_t>(as(std::uint64_t, x"
             << src.idx() << ")));\n";
    }

    template <unsigned NumLanes>
    void fadd(GenericVReg<4, NumLanes> dst, GenericVReg<4, NumLanes> src1,
              GenericVReg<4, NumLanes> src2)
    {
        static_assert(NumLanes == 1 || NumLanes == 2 || NumLanes == 4);

        for (unsigned i = 0; i < NumLanes; ++i)
        {
            oss_ << "  " << to_string(dst[i]) << " = " << to_string(src1[i])
                 << " + " << to_string(src2[i]) << ";";
        }

        oss_ << "\n";
    }

    void faddp(GenericVReg<4, 4> dst, GenericVReg<4, 4> src1,
               GenericVReg<4, 4> src2)
    {
        oss_ << "  " << to_string(dst[0]) << " = " << to_string(src1[0])
             << " + " << to_string(src2[2]) << "; ";
        oss_ << "  " << to_string(dst[1]) << " = " << to_string(src1[1])
             << " + " << to_string(src2[3]) << ";\n";
    }

    void faddp(GenericVReg<4, 1> dst, GenericVReg<4, 2> src)
    {
        oss_ << "  " << to_string(dst[0]) << " = " << to_string(src[0]) << " + "
             << to_string(src[1]) << ";\n";
    }

    template <unsigned NumLanes>
    void fmla(GenericVReg<4, NumLanes> vd, GenericVReg<4, NumLanes> va,
              GenericVReg<4, NumLanes> vb)
    {
        oss_ << "  for (unsigned i = 0; i < " << NumLanes << "; ++i) vmm"
             << vd.idx() << "[i] += vmm" << va.idx() << "[i] * vmm" << vb.idx()
             << "[i];\n";
    }

    template <unsigned NumLanes>
    void fmax(GenericVReg<4, NumLanes> vd, GenericVReg<4, NumLanes> va,
              GenericVReg<4, NumLanes> vb)
    {
        oss_ << "  for (unsigned i = 0; i < " << NumLanes << "; ++i) vmm"
             << vd.idx() << "[i] = std::max(vmm" << va.idx() << "[i] * vmm"
             << vb.idx() << "[i]);\n";
    }

    template <unsigned NumLanes>
    void fmla(GenericVReg<4, NumLanes> vd, GenericVReg<4, NumLanes> va,
              GenericVRegLane<4> vb)
    {
        oss_ << "  for (unsigned i = 0; i < " << NumLanes << "; ++i) vmm"
             << vd.idx() << "[i] += vmm" << va.idx() << "[i] * "
             << to_string(vb) << ";\n";
    }

    template <unsigned NumLanes>
    void mov(GenericVReg<4, NumLanes> dst, GenericVReg<4, NumLanes> src)
    {
        oss_ << "  for (unsigned i = 0; i < " << NumLanes << "; ++i) vmm"
             << dst.idx() << "[i] = vmm" << src.idx() << "[i];\n";
    }

    void ld1(GenericVRegLane<4> dst, ptr_impl p)
    {
        oss_ << "  " << to_string(dst) << " = as(float*, x" << p.reg.idx()
             << ")[0];\n";
    }

    void st1(GenericVRegLane<4> dst, ptr_impl p)
    {
        oss_ << "  as(float*, x" << p.reg.idx() << ")[0] = " << to_string(dst)
             << ";\n";
    }

    template <unsigned NumLanes>
    void ldr(GenericVReg<4, NumLanes> dst, ptr_impl p)
    {
        oss_ << "  for (unsigned i = 0; i < " << NumLanes << "; ++i) vmm"
             << dst.idx() << "[i]"
             << " = as(float*, x" << p.reg.idx() << ")[i];\n";
    }

    template <unsigned NumLanes>
    void ldp(GenericVReg<4, NumLanes> d1, GenericVReg<4, NumLanes> d2,
             ptr_impl p)
    {
        oss_ << "  for (unsigned i = 0; i < " << NumLanes << "; ++i) vmm"
             << d1.idx() << "[i]"
             << " = as(float*, x" << p.reg.idx() << ")[i];\n";
        oss_ << "  for (unsigned i = 0; i < " << NumLanes << "; ++i) vmm"
             << d2.idx() << "[i]"
             << " = as(float*, x" << p.reg.idx() << ")[i + " << NumLanes
             << "];\n";
    }

    template <unsigned NumLanes>
    void ldr(GenericVReg<4, NumLanes> dst, pre_ptr_impl p)
    {
        oss_ << "  x" << p.reg.idx() << " += " << p.pre << ";\n";
        ldr(dst, ptr_impl{p.reg});
    }

    template <unsigned NumLanes>
    void ldr(GenericVReg<4, NumLanes> dst, post_ptr_impl p)
    {
        ldr(dst, ptr_impl{p.reg});
        oss_ << "  x" << p.reg.idx() << " += " << p.post << ";\n";
    }

    template <unsigned NumLanes>
    void ldp(GenericVReg<4, NumLanes> d1, GenericVReg<4, NumLanes> d2,
             pre_ptr_impl p)
    {
        oss_ << "  x" << p.reg.idx() << " += " << p.pre << ";\n";
        ldp(d1, d2, ptr_impl{p.reg});
    }

    template <unsigned NumLanes>
    void ldp(GenericVReg<4, NumLanes> d1, GenericVReg<4, NumLanes> d2,
             post_ptr_impl p)
    {
        ldp(d1, d2, ptr_impl{p.reg});
        oss_ << "  x" << p.reg.idx() << " += " << p.post << ";\n";
    }

    template <unsigned NumLanes>
    void meta_ldp_post_ptr(GenericVReg<4, NumLanes> d1,
                           GenericVReg<4, NumLanes> d2, Reg64 reg, int delta)
    {
        ldp(d1, d2, ptr_impl{reg});
        oss_ << "  x" << reg.idx() << " += " << delta << ";\n";
    }

    template <unsigned NumLanes>
    void meta_ldr_post_ptr(GenericVReg<4, NumLanes> d1, Reg64 reg, int delta)
    {
        ldr(d1, ptr_impl{reg});
        oss_ << "  x" << reg.idx() << " += " << delta << ";\n";
    }

    void meta_ldr_post_ptr(Reg32 dst, Reg64 reg, int delta)
    {
        oss_ << "  x" << dst.idx() << " = as(std::uint32_t*, x" << reg.idx()
             << ")[0];\n";
        oss_ << "  x" << reg.idx() << " += " << delta << ";\n";
    }

    void meta_ldr_post_ptr(Reg64 dst, Reg64 reg, int delta)
    {
        oss_ << "  x" << dst.idx() << " = as(std::uint64_t*, x" << reg.idx()
             << ")[0];\n";
        oss_ << "  x" << reg.idx() << " += " << delta << ";\n";
    }

    template <unsigned NumLanes>
    void str(GenericVReg<4, NumLanes> src, ptr_impl p)
    {
        oss_ << "  for (unsigned i = 0; i < " << NumLanes
             << "; ++i) as(float*, x" << p.reg.idx() << ")[i] = vmm"
             << src.idx() << "[i];\n";
    }

    template <unsigned NumLanes>
    void stp(GenericVReg<4, NumLanes> s1, GenericVReg<4, NumLanes> s2,
             ptr_impl p)
    {
        oss_ << "  for (unsigned i = 0; i < " << NumLanes
             << "; ++i) as(float*, x" << p.reg.idx() << ")[i] = vmm" << s1.idx()
             << "[i];   ";
        oss_ << "  for (unsigned i = 0; i < " << NumLanes
             << "; ++i) as(float*, x" << p.reg.idx() << ")[i + " << NumLanes
             << "] = vmm" << s2.idx() << "[i];\n";
    }

    template <unsigned NumLanes>
    void str(GenericVReg<4, NumLanes> src, pre_ptr_impl p)
    {
        oss_ << "  x" << p.reg.idx() << " += " << p.pre << ";\n";
        str(src, ptr_impl{p.reg});
    }

    template <unsigned NumLanes>
    void str(GenericVReg<4, NumLanes> src, post_ptr_impl p)
    {
        str(src, ptr_impl{p.reg});
        oss_ << "  x" << p.reg.idx() << " += " << p.post << ";\n";
    }

    template <unsigned NumLanes>
    void stp(GenericVReg<4, NumLanes> s1, GenericVReg<4, NumLanes> s2,
             pre_ptr_impl p)
    {
        oss_ << "  x" << p.reg.idx() << " += " << p.pre << ";\n";
        stp(s1, s2, ptr_impl{p.reg});
    }

    template <unsigned NumLanes>
    void stp(GenericVReg<4, NumLanes> s1, GenericVReg<4, NumLanes> s2,
             post_ptr_impl p)
    {
        stp(s1, s2, ptr_impl{p.reg});
        oss_ << "  x" << p.reg.idx() << " += " << p.post << ";\n";
    }

    void meta_sadd_imm(Reg64 r, std::int64_t val)
    {
        oss_ << "  x" << r.idx() << " += " << val << ";\n";
    }

    void meta_add_imm(Reg64 r, int val)
    {
        oss_ << "  x" << r.idx() << " += " << val << ";\n";
    }

    void meta_mov_imm(Reg64 r, std::int64_t val)
    {
        oss_ << "  x" << r.idx() << " = " << val << ";\n";
    }

    void meta_sub_imm(Reg64 r, int val)
    {
        oss_ << "  x" << r.idx() << " -= " << val << ";\n";
    }

    std::shared_ptr<std::string> make_label()
    {
        return std::make_shared<std::string>(std::string("locallabel") +
                                             std::to_string(label_idx_++));
    }

    void cbnz(Reg64 reg, std::string const& label)
    {
        oss_ << "  if (x" << reg.idx() << " != 0) goto " << label << ";\n";
    }

    enum struct comparison
    {
        LT = 1,
        NE = 2
    };

    void b(std::string const& label) { oss_ << "  goto " << label << ";\n"; }

    void b(comparison c, std::string const& label)
    {
        if (c == comparison::LT)
        {
            oss_ << "  if (cmp_lhs < cmp_rhs) goto " << label << ";\n";
        }
        if (c == comparison::NE)
        {
            oss_ << "  if (cmp_lhs != cmp_rhs) goto " << label << ";\n";
        }
    }

    void L_aarch64(std::string const& l) { oss_ << l << ":\n"; }

    void meta_cmp(Reg64 reg, std::int64_t val)
    {
        oss_ << "  cmp_lhs = x" << reg.idx() << ";  cmp_rhs = " << val << ";\n";
    }

    void cmp(Reg64 reg, int val)
    {
        oss_ << "  cmp_lhs = x" << reg.idx() << ";  cmp_rhs = " << val << ";\n";
    }

    void tbnz(Reg64 reg, int bit, std::string const& label)
    {
        oss_ << "  if ((x" << reg.idx() << " >> " << bit << ") & 1) goto "
             << label << ";\n";
    }

private:
    std::atomic<int> label_idx_{0};
};

// } // namespace Xbyak
} // namespace tensillica
} // namespace dabun
