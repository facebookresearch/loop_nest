#pragma once

namespace facebook
{
namespace sysml
{
namespace aot
{

struct avx2
{
};
struct avx512
{
};
struct avx2_plus
{
};
struct aarch64
{
};

template <class>
struct isa_traits;

template <>
struct isa_traits<avx2>
{
    static constexpr int total_vector_registers = 16;
    static constexpr int vector_register_mask   = 1;
    static constexpr int vector_size            = 8;
};

template <>
struct isa_traits<avx512>
{
    static constexpr int total_vector_registers = 32;
    static constexpr int vector_register_mask   = 0;
    static constexpr int vector_size            = 16;
};

template <>
struct isa_traits<avx2_plus>
{
    static constexpr int total_vector_registers = 16;
    static constexpr int vector_register_mask   = 0;
    static constexpr int vector_size            = 8;
};

template <>
struct isa_traits<aarch64>
{
    static constexpr int total_vector_registers = 32;
    static constexpr int vector_register_mask   = 0;
    static constexpr int vector_size            = 4;
};

} // namespace aot
} // namespace sysml
} // namespace facebook
