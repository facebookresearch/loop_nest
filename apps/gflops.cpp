#include "dabun/peak_gflops.hpp"
#include <iostream>

using namespace dabun;

int main()
{
    std::cout << peak_gflops(DABUN_ISA (), 1000000) << "\n";
    // std::cout << fn() << "\n";
}
