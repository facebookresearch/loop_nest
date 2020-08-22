#pragma once

#pragma once

#if !defined(ARM_LOOP_NEST)

#include "elementwise_operation.h"

#else

#include "isa.h"

namespace facebook
{
namespace sysml
{
namespace aot
{

template <class ISA>
class elementwise_operation
{
private:
    bool is_relu_ = false;

public:
    bool is_relu() const { return is_relu_; }

    explicit elementwise_operation(bool b)
        : is_relu_(b)
    {
    }
};

inline auto elementwise_relu =
    std::make_shared<elementwise_operation<aarch64>>(true);

inline auto elementwise_bias =
    std::make_shared<elementwise_operation<aarch64>>(false);

inline auto elementwise_multiply =
    std::make_shared<elementwise_operation<aarch64>>(false);

} // namespace aot
} // namespace sysml
} // namespace facebook

#endif
