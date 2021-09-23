#pragma once

// template <class Float>
// void apply_relu(Float* Begin, Float* End)
// {
//     for (; Begin != End; ++Begin)
//     {
//         if constexpr (std::is_same_v<Float, fp16>)
//         {
//             *Begin = static_cast<fp16>(
//                 std::max(static_cast<float>(0), static_cast<float>(*Begin)));
//         }
//         else
//         {
//             *Begin = std::max(static_cast<Float>(0), *Begin);
//         }
//     }
// }
