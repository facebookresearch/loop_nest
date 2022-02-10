#pragma once

#include <memory>

namespace dabun
{
namespace arm
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

template <class ISA>
class relu_elementwise_operation
{
};

template <class ISA>
class single_tensor_elementwise_operation
{
};

template <class T>
inline auto elementwise_relu = std::make_shared<elementwise_operation<T>>(true);

template <class T>
inline auto
    elementwise_bias = std::make_shared<elementwise_operation<T>>(false);

template <class T>
inline auto
    elementwise_multiply = std::make_shared<elementwise_operation<T>>(false);

} // namespace arm
} // namespace dabun
