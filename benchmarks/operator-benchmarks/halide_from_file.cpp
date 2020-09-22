#include "loop_nest.h"
#include "serialization.h"
#include "translate_to_halide.h"
#include "utils.h"

#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#ifndef DABUN_ISA
#define DABUN_ISA avx2
#endif

std::int64_t compute_size(std::map<std::string, int> sizes,
                          std::map<std::string, int> strides)
{
    std::int64_t size = 1;

    for (auto const& s : sizes)
    {
        if (strides.count(s.first))
            size += (s.second - 1) * strides.at(s.first);
    }
    return size;
}

std::int64_t compute_flops(std::map<std::string, int> const& sizes,
                           std::map<std::string, int> const& C_strides,
                           std::map<std::string, int> const& A_strides,
                           std::map<std::string, int> const& B_strides)
{
    std::int64_t flops = 2;
    for (auto const& s : sizes)
    {
        if (C_strides.count(s.first) || B_strides.count(s.first) ||
            A_strides.count(s.first))
            flops *= s.second;
    }
    return flops;
}

void help() { std::cout << "Usage: <serialized-benchmark-path>" << std::endl; }

int main(int argc, char* argv[])
{
    using facebook::sysml::aot::aot_fn_cast;
    using facebook::sysml::aot::avx2;
    using facebook::sysml::aot::avx2_plus;
    using facebook::sysml::aot::avx512;

    if (argc != 2)
    {
        help();
        return 1;
    }

    std::string file_path = argv[1];

    auto serialized =
        facebook::sysml::aot::serialized_loop_nest_inputs::from_file(file_path);

    std::cout << "Benchmark: " << file_path << std::endl;

    auto order        = serialized.get_order();
    auto sizes        = serialized.get_sizes();
    auto C_formula    = serialized.get_formula("C");
    auto A_formula    = serialized.get_formula("A");
    auto B_formula    = serialized.get_formula("B");
    auto C_strides    = serialized.get_strides("C");
    auto A_strides    = serialized.get_strides("A");
    auto B_strides    = serialized.get_strides("B");
    auto unroll_limit = serialized.get_unroll_limit();

    auto fn = facebook::sysml::aot::LoopNestToHalide<DABUN_ISA>(
        order, sizes, C_formula, A_formula, B_formula, C_strides, A_strides,
        B_strides, unroll_limit);

    auto compile      = [&fn]() { fn.compile_jit(); };
    auto compile_secs = measureFastestWithWarmup(compile, 0, 1);
    std::cout << "Compile: " << compile_secs << std::endl;

    auto A = getRandomVector<float>(compute_size(sizes, A_strides));
    auto B = getRandomVector<float>(compute_size(sizes, B_strides));
    auto C = getRandomVector<float>(compute_size(sizes, C_strides));

    auto secs = measureFastestWithWarmup(
        [&]() { fn.run_on_aligned_data(C.data(), A.data(), B.data()); }, 10,
        1000);

    double gflops =
        (1.0 * compute_flops(sizes, C_strides, A_strides, B_strides)) /
        1000000000;

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
}
