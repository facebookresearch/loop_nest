#pragma once

#if !defined(ARM_LOOP_NEST)

#include "transposer.h"

#else

#include "code_generator.h"
#include "common.h"
#include "isa.h"
#include "log.h"
#include "math.h"

#include <cassert>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace facebook
{
namespace sysml
{
namespace aot
{

template <class ISA>
class transposer_jitter
    : public code_generator<void(float* Out, float const* In)>
{

public:
    transposer_jitter(std::vector<std::pair<std::string, int>> const&,
                      std::map<std::string, int> const&,
                      std::map<std::string, int> const&,
                      std::map<std::string, int> const&,
                      std::optional<int> = std::nullopt)

    {
        ret();
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook

#endif
