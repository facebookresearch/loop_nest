#include "isa_bench.h"
#include <iostream>

using namespace facebook::sysml;

int main()
{
    std::cout << peak_gflops(aot:: CT_ISA (), 1000000) << "\n";
    // std::cout << fn() << "\n";
}
