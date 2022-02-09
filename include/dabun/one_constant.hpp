#pragma once

namespace dabun
{

template <class Float>
Float const one_actual_constant = static_cast<Float>(1);

template <class Float>
Float const* const one_constant = &one_actual_constant<Float>;

} // namespace dabun
