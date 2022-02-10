#include "baselines.h"
#include "loop_nest.h"
#include "loop_nest_baseline.h"
#include "one_constant.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifndef DABUN_ISA
#define DABUN_ISA avx2
#endif

#ifndef PLUS_OP
#define PLUS_OP basic_plus
#endif

#ifndef MULTIPLIES_OP
#define MULTIPLIES_OP basic_multiplies
#endif

int main(int argc, char *argv[]) {
  using facebook::sysml::aot::aot_fn_cast;
  using facebook::sysml::aot::avx2;
  using facebook::sysml::aot::avx512;

  using facebook::sysml::aot::basic_multiplies;
  using facebook::sysml::aot::basic_plus;
  using facebook::sysml::aot::max;
  using facebook::sysml::aot::min;
  using facebook::sysml::aot::operation_pair;
  using facebook::sysml::aot::operation_pair_base;

  {
    std::string size_str = argv[1];
    int size = std::stoi(size_str);

    int ArCr = size;
    int AcBr = size;
    int BcCc = size;

    std::vector<std::pair<std::string, int>> schedule = {
        {"AcBr", 256}, {"BcCc", 16}, {"ArCr", 3},
        {"AcBr", 1},   {"ArCr", 1},  {"BcCc", 1}};

    std::shared_ptr<operation_pair_base> op_pair =
        std::make_shared<operation_pair<PLUS_OP, MULTIPLIES_OP>>();

    auto gen_loop_nest = [&]() {
      return facebook::sysml::aot::loop_nest_code_generator<DABUN_ISA>(
                 schedule, {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
                 {"ArCr", "BcCc"}, {"ArCr", "AcBr"}, {"AcBr", "BcCc"},
                 {{"ArCr", BcCc}, {"BcCc", 1}}, {{"ArCr", AcBr}, {"AcBr", 1}},
                 {{"AcBr", BcCc}, {"BcCc", 1}}, op_pair, 1024)
          .get_unique();
    };

    auto compile_secs = measureFastestWithWarmup(gen_loop_nest, 0, 1);
    std::cout << "Compile: " << compile_secs << std::endl;

    auto fnx = gen_loop_nest();
    auto fny = aot_fn_cast<void(int)>(std::move(fnx));
    auto fn = aot_fn_cast<void(float *, float const *, float const *, int)>(
        std::move(fny));

    auto A = get_random_vector<float>(AcBr * ArCr);
    auto B = get_random_vector<float>(AcBr * BcCc);

    auto CN = get_random_vector<float>(ArCr * BcCc);
    auto CJ = CN;

    fn(CJ.data(), A.data(), B.data(), 0);

    auto secs = measureFastestWithWarmup(
        [&]() { fn(CJ.data(), A.data(), B.data(), 0); }, 10, 1000);

    double gflops = 1.0 * AcBr * ArCr * BcCc * 2 / 1000000000;

    std::cout << "GFLOPS: " << (gflops / secs) << "\n";
  }
}
