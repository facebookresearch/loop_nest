#pragma once

#include "dabun/core.hpp"
#include "dabun/xbyak.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

namespace dabun
{

static inline constexpr int skip_postop = 0b10;
static inline constexpr int alpha_1     = 0b01;
static inline constexpr int alpha_0     = 0b00;

enum access_kind
{
    SCALAR,
    VECTOR_PACKED,
    VECTOR_STRIDED
};

inline std::string to_string(access_kind akind)
{
    switch (akind)
    {
    case SCALAR:
        return "scalar";
    case VECTOR_PACKED:
        return "vector_packed";
    case VECTOR_STRIDED:
        return "vector_strided";
    }
    return "unknown";
}

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
    Xbyak::Reg64  reg = Xbyak::Reg64(0);
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
        return std::tie(offset, mask, traits->name) <
               std::tie(o.offset, mask, o.traits->name);
    }

    bool operator==(memory_argument_type const& o) const
    {
        return std::tie(offset, mask, traits->name) ==
               std::tie(o.offset, mask, o.traits->name);
    }

    std::string readable() const
    {
        assert(traits);
        return traits->name + "[" + std::to_string(offset) + ":" +
               std::to_string(traits->access == SCALAR ? 1 : vector_size) +
               "]{" + std::to_string(traits->innermost_stride) + "}{" +
               std::to_string(mask) + "}";
    }
};

struct in_register_tensor_pointer_type
{
    std::string                name;
    Xbyak::Reg64               reg;
    std::map<std::string, int> strides;
};

inline int get_cursor_offset(std::map<std::string, int> coordinates,
                             std::map<std::string, int> strides)
{
    int off = 0;
    for (auto const& s : strides)
    {
        off += coordinates[s.first] * s.second;
    }
    return off;
}

} // namespace dabun
