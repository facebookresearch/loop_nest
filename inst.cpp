#include "code_generator.h"
#include "math.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

class test : public facebook::sysml::aot::code_generator<std::int64_t()>
{
public:
    test() {
        vfmadd231ps(zmm3, zmm0, ptr_b[rcx + 0xc]);
        ret();
    }
};

int main()
{
    auto fn = test().get_shared();
    fn.save_to_file("zi.asm");
    // std::cout << fn() << "\n";
}
