#pragma once

template <class Float>
void apply_relu(Float* Begin, float* End)
{
    for (; Begin != End; ++Begin)
    {
        *Begin = std::max(static_cast<Float>(0), *Begin);
    }
}
