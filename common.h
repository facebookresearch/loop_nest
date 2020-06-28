#pragma once

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#include <cassert>
#include <map>
#include <string>
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
    int                  offset;
    tensor_traits const* traits;
    int                  mask;

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

} // namespace aot
} // namespace sysml
} // namespace facebook
