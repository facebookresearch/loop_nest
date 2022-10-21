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

namespace dabun::tests::baseline
{

std::function<void(
    float*, float const*, float const*,
    int)> inline slow_loop_nest_baseline(std::vector<std::pair<std::string,
                                                               int>> const&
                                             order,
                                         std::map<std::string, int> const&
                                                                      sizes,
                                         std::set<std::string> const& C_formula,
                                         std::set<
                                             std::string> const& /*A_formula*/,
                                         std::set<
                                             std::string> const& /*B_formula*/,
                                         std::map<std::string, int> const&
                                             C_strides,
                                         std::map<std::string, int> const&
                                             A_strides,
                                         std::map<std::string, int> const&
                                              B_strides,
                                         bool perform_relu = false)
{
    return [=](float* Cptr, float const* Aptr, float const* Bptr, int alpha)
    {
        std::map<std::string, std::vector<int>> limits;
        for (auto const& s : sizes)
        {
            limits[s.first].push_back(s.second);
        }

        if (alpha == 0)
        {
            std::function<void(float*, int)> recursive_zeroing =
                [&](float* C, int order_depth)
            {
                if (order_depth == order.size())
                {
                    C[0] = 0.f;
                }
                else
                {
                    if (C_formula.count(order.at(order_depth).first) > 0)
                    {
                        auto var   = order.at(order_depth).first;
                        auto limit = limits[var].back();
                        auto delta = order.at(order_depth).second;
                        int  full  = limit / delta;
                        int  rest  = limit % delta;

                        limits[var].push_back(delta);
                        for (int i = 0; i < full; ++i)
                        {
                            recursive_zeroing(C, order_depth + 1);
                            C += C_strides.count(var)
                                     ? C_strides.at(var) * delta
                                     : 0;
                        }
                        limits[var].pop_back();

                        if (rest)
                        {
                            limits[var].push_back(rest);
                            recursive_zeroing(C, order_depth + 1);
                            limits[var].pop_back();
                        }
                    }
                    else
                    {
                        recursive_zeroing(C, order_depth + 1);
                    }
                }
            };

            recursive_zeroing(Cptr, 0);
        }

        std::function<void(float*, float const*, float const*, int)>
            recursive_compute =
                [&](float* C, float const* A, float const* B, int order_depth)
        {
            if (order_depth == order.size())
            {
                C[0] += A[0] * B[0];
            }
            else
            {
                auto var   = order.at(order_depth).first;
                auto limit = limits[var].back();
                auto delta = order.at(order_depth).second;
                int  full  = limit / delta;
                int  rest  = limit % delta;

                limits[var].push_back(delta);
                for (int i = 0; i < full; ++i)
                {
                    recursive_compute(C, A, B, order_depth + 1);
                    C += C_strides.count(var) ? C_strides.at(var) * delta : 0;
                    A += A_strides.count(var) ? A_strides.at(var) * delta : 0;
                    B += B_strides.count(var) ? B_strides.at(var) * delta : 0;
                }
                limits[var].pop_back();

                if (rest)
                {
                    limits[var].push_back(rest);
                    recursive_compute(C, A, B, order_depth + 1);
                    limits[var].pop_back();
                }
            }
        };

        recursive_compute(Cptr, Aptr, Bptr, 0);

        if (perform_relu)
        {
            std::function<void(float*, int)> recursive_relu =
                [&](float* C, int order_depth)
            {
                if (order_depth == order.size())
                {
                    C[0] = std::max(C[0], 0.f);
                }
                else
                {
                    if (C_formula.count(order.at(order_depth).first) > 0)
                    {
                        auto var   = order.at(order_depth).first;
                        auto limit = limits[var].back();
                        auto delta = order.at(order_depth).second;
                        int  full  = limit / delta;
                        int  rest  = limit % delta;

                        limits[var].push_back(delta);
                        for (int i = 0; i < full; ++i)
                        {
                            recursive_relu(C, order_depth + 1);
                            C += C_strides.count(var)
                                     ? C_strides.at(var) * delta
                                     : 0;
                        }
                        limits[var].pop_back();

                        if (rest)
                        {
                            limits[var].push_back(rest);
                            recursive_relu(C, order_depth + 1);
                            limits[var].pop_back();
                        }
                    }
                    else
                    {
                        recursive_relu(C, order_depth + 1);
                    }
                }
            };

            recursive_relu(Cptr, 0);
        }
    };
}

// Slightly optimized version for faster testing (10x faster)
std::function<void(
    float*, float const*, float const*,
    int)> inline loop_nest_baseline(std::vector<std::pair<std::string,
                                                          int>> const& order,
                                    std::map<std::string, int> const&  sizes,
                                    std::set<std::string> const& C_formula,
                                    std::set<std::string> const& /*A_formula*/,
                                    std::set<std::string> const& /*B_formula*/,
                                    std::map<std::string, int> const& C_strides,
                                    std::map<std::string, int> const& A_strides,
                                    std::map<std::string, int> const& B_strides,
                                    bool perform_relu = false)
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
    std::vector<int> order_C_strides(order.size());
    std::vector<int> order_A_strides(order.size());
    std::vector<int> order_B_strides(order.size());

    for (int i = 0; i < order.size(); ++i)
    {
        order_ids[i]   = var_to_id[order[i].first];
        order_delta[i] = order[i].second;

        order_C_strides[i] = C_strides.count(order[i].first)
                                 ? C_strides.at(order[i].first) * order_delta[i]
                                 : 0;
        order_A_strides[i] = A_strides.count(order[i].first)
                                 ? A_strides.at(order[i].first) * order_delta[i]
                                 : 0;
        order_B_strides[i] = B_strides.count(order[i].first)
                                 ? B_strides.at(order[i].first) * order_delta[i]
                                 : 0;
    }

    std::vector<int> init_len;
    std::vector<int> init_C_strides;

    for (auto const& s : sizes)
    {
        if (C_formula.count(s.first) > 0)
        {
            init_len.push_back(s.second);
            init_C_strides.push_back(C_strides.at(s.first));
        }
    }

    return [=](float* Cptr, float const* Aptr, float const* Bptr, int alpha)
    {
        auto limits = initial_limits;

        if (alpha == 0)
        {
            std::function<void(float*, int)> recursive_zeroing =
                [&](float* C, int init_depth)
            {
                if (init_depth == init_len.size())
                {
                    C[0] = 0.f;
                }
                else
                {
                    for (int i = 0; i < init_len[init_depth]; ++i)
                    {
                        recursive_zeroing(C, init_depth + 1);
                        C += init_C_strides[init_depth];
                    }
                }
            };

            recursive_zeroing(Cptr, 0);
        }

        std::function<void(float*, float const*, float const*, int)>
            recursive_compute =
                [&](float* C, float const* A, float const* B, int order_depth)
        {
            if (order_depth == order_ids.size())
            {
                C[0] += A[0] * B[0];
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
                    recursive_compute(C, A, B, order_depth + 1);
                    C += order_C_strides[order_depth];
                    A += order_A_strides[order_depth];
                    B += order_B_strides[order_depth];
                }
                limits[var] = save;

                if (rest)
                {
                    int s = std::exchange(limits[var], rest);
                    recursive_compute(C, A, B, order_depth + 1);
                    limits[var] = s;
                }
            }
        };

        recursive_compute(Cptr, Aptr, Bptr, 0);

        if (perform_relu)
        {
            std::function<void(float*, int)> recursive_relu =
                [&](float* C, int init_depth)
            {
                if (init_depth == init_len.size())
                {
                    C[0] = std::max(C[0], 0.f);
                }
                else
                {
                    for (int i = 0; i < init_len[init_depth]; ++i)
                    {
                        recursive_relu(C, init_depth + 1);
                        C += init_C_strides[init_depth];
                    }
                }
            };

            recursive_relu(Cptr, 0);
        }
    };
}

} // namespace dabun::tests::baseline
