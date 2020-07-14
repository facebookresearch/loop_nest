#pragma once

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#include <cassert>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

namespace facebook
{
namespace sysml
{
namespace aot
{

enum access_kind
{
    SCALAR,
    VECTOR_PACKED,
    VECTOR_STRIDED
};

struct loop_descriptor
{
    std::string var;
    int         end;
    int         delta;
};

struct tensor_traits
{
    std::string   name;
    access_kind   access;
    Xbyak::Reg64  reg;
    Xbyak::Label* stridesLabel;
    int           innermost_stride;
    int           access_len;
};

template <int vector_size>
struct memory_argument_type
{
    int                        offset;
    tensor_traits const*       traits;
    int                        mask;
    std::map<std::string, int> coordinates;

    memory_argument_type(int offset, tensor_traits const* traits, int mask,
                         std::map<std::string, int> coordinates = {})
        : offset(offset)
        , traits(traits)
        , mask(mask)
        , coordinates(coordinates){};
    // We are not comparing the mask

    bool operator<(memory_argument_type const& o) const
    {
        return std::tie(offset, traits->name) <
               std::tie(o.offset, o.traits->name);
    }

    bool operator==(memory_argument_type const& o) const
    {
        return std::tie(offset, traits->name) ==
               std::tie(o.offset, o.traits->name);
    }

    std::string readable() const
    {
        assert(traits);
        return traits->name + "[" + std::to_string(offset) + ":" +
               std::to_string(traits->access == SCALAR ? 1 : vector_size) +
               "]{" + std::to_string(traits->innermost_stride) + "}";
    }
};

struct in_register_tensor_pointer_type
{
    std::string                name;
    Xbyak::Reg64               reg;
    std::map<std::string, int> strides;
};

int get_cursor_offset(std::map<std::string, int> coordinates,
                      std::map<std::string, int> strides)
{
    int off = 0;
    for (auto const& s : strides)
    {
        off += coordinates[s.first] * s.second;
    }
    return off;
}

// Sourced from https://en.cppreference.com/w/cpp/numeric/bit_cast
// to enable bit_cast from C++20
template <class To, class From>
typename std::enable_if_t<sizeof(To) == sizeof(From) &&
                              std::is_trivially_copyable_v<From> &&
                              std::is_trivially_copyable_v<To>,
                          To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept
{
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation additionally requires destination type "
                  "to be trivially constructible");

    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

} // namespace aot
} // namespace sysml
} // namespace facebook
