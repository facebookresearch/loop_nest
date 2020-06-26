#pragma once

#include "code_generator.h"
#include "isa.h"
#include "log.h"

#include <cassert>
#include <map>
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
private:
    static constexpr int vector_size = isa_traits<ISA>::vector_size;

    using mask_register_type =
        std::conditional_t<std::is_same_v<ISA, avx2>, Ymm, OpMask>;
    using Vmm = std::conditional_t<std::is_same_v<ISA, avx512>, Zmm, Ymm>;

private:
    Reg64 out_reg  = rdi;
    Reg64 in_reg   = rsi;
    Reg64 loop_reg = rax;

    Label mask_label;

    std::vector<std::pair<std::string, int>> order;
    std::map<std::string, int>               sizes;
    std::map<std::string, int>               out_strides;
    std::map<std::string, int>               in_strides;

    std::string vectorized_var;

    // Used vector (and mask registers)
    mask_register_type full_mask, tail_mask, in_temp_mask, out_temp_mask;

    // Possible regs that store the gather/scatter strides
    Vmm in_access_strides_reg, out_access_strides_reg;

    // In/out data through the vector reg
    Vmm in_out_vmm_reg;

    void check_representation()
    {
        // Make sure strides (and sizes) exist for each order variable
        for (auto const& o : order)
        {
            assert(out_strides.count(o.first) > 0);
            assert(in_strides.count(o.first) > 0);
            assert(sizes.count(o.first) > 0);
        }

        std::map<std::string, int> last_step;

        for (auto const& o : order)
        {
            if (last_step.count(o.first))
            {
                assert(last_step[o.first] >= o.second &&
                       "The steps in 'order' need to be non-increasing");
            }
            last_step[o.first] = o.second;
        }

        for (auto const& o : order)
        {
            assert(last_step[o.first] == 1 &&
                   "Last step in order not equal to 1");
        }

        for (auto const& o : order)
        {
            if (o.first == vectorized_var)
            {
                assert(o.second == 1 || (o.second % vector_size == 0));
            }
        }
    }

public:
    transposer_jitter(std::vector<std::pair<std::string, int>> const& Order,
                      std::map<std::string, int> const&               Sizes,
                      std::map<std::string, int> const& Out_strides,
                      std::map<std::string, int> const& In_strides)
        : order(Order)
        , sizes(Sizes)
        , out_strides(Out_strides)
        , in_strides(In_strides)
    {
        assert(order.size());
        vectorized_var = order.back().first;

        check_representation();
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
