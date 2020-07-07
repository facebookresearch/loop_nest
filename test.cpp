#include "Halide.h"
#include <iostream>
#include <vector>

using namespace Halide;

int main() {
  Var i("i"), j("j");
  RDom k(0, 2);

  std::vector<int> A_data = {1,2,3,4};
  std::vector<int> B_data = {10, 20, 30, 40};
  Buffer<int> A(A_data.data(), {2, 2}, "A");
  Buffer<int> B(B_data.data(), {2, 2}, "B");

  Func C("C");
  C(j, i) = 0;
  C(j, i) = 0.5 * C(j, i) + A(k, i) * B(j, k);

  C.print_loop_nest();
  Buffer<int> C_result = C.realize(2, 2);
  for(auto const& e : C_result) {
    std::cout << e << std::endl;
  }

}
