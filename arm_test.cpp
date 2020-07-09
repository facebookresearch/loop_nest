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

class test : public facebook::sysml::aot::code_generator<std::int64_t(
                 std::int64_t, std::int64_t)>
{
public:
    test()
    {
        add(w0, w1, w0);
        ret();
    }
};

int main()
{
    auto fn = test().get_shared();
    fn.save_to_file("zi.asm");
    std::cout << fn(3, 8) << "\n";
}
