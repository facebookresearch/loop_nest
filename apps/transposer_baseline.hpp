// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dabun
{

template <class Arithmetic>
auto transposer_baseline(std::vector<std::pair<std::string, int>> const& order,
                         std::map<std::string, int> const&               sizes,
                         std::map<std::string, int> const& out_strides,
                         std::map<std::string, int> const& in_strides)
{
    // Just optimizing out the map lookups.

    std::map<std::string, int> var_to_id;

    int next = 0;
    for (auto const& s : sizes)
    {
        var_to_id[s.first] = next++;
    }

    std::vector<int> initial_limits(next);

    for (auto const& s : sizes)
    {
        initial_limits[var_to_id[s.first]] = s.second;
    }

    std::vector<int> order_ids(order.size());
    std::vector<int> order_delta(order.size());
    std::vector<int> order_in_strides(order.size());
    std::vector<int> order_out_strides(order.size());

    for (int i = 0; i < order.size(); ++i)
    {
        order_ids[i]   = var_to_id[order[i].first];
        order_delta[i] = order[i].second;

        order_in_strides[i] =
            in_strides.count(order[i].first)
                ? in_strides.at(order[i].first) * order_delta[i]
                : 0;
        order_out_strides[i] =
            out_strides.count(order[i].first)
                ? out_strides.at(order[i].first) * order_delta[i]
                : 0;
    }

    return [=](Arithmetic* out_ptr, Arithmetic const* in_ptr) {
        auto limits = initial_limits;

        std::function<void(Arithmetic*, Arithmetic const*, int)>
            recursive_compute =
                [&](Arithmetic* out, Arithmetic const* in, int order_depth) {
                    if (order_depth == order_ids.size())
                    {
                        out[0] = in[0];
                    }
                    else
                    {
                        auto var   = order_ids[order_depth];
                        auto delta = order_delta[order_depth];
                        auto limit = limits[var];
                        auto full  = limit / delta;
                        auto rest  = limit % delta;

                        auto save = std::exchange(limits[var], delta);
                        for (int i = 0; i < full; ++i)
                        {
                            recursive_compute(out, in, order_depth + 1);
                            in += order_in_strides[order_depth];
                            out += order_out_strides[order_depth];
                        }
                        limits[var] = save;

                        if (rest)
                        {
                            int s = std::exchange(limits[var], rest);
                            recursive_compute(out, in, order_depth + 1);
                            limits[var] = s;
                        }
                    }
                };

        recursive_compute(out_ptr, in_ptr, 0);
    };
}

} // namespace dabun
