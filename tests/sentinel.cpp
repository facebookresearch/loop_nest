#include "dabun/thread/operating_cpu_set.hpp"

#include <catch2/catch.hpp>
#include <iostream>

int Factorial(int number)
{
    // return number <= 1 ? number : Factorial(number - 1) * number; // fail
    return number <= 1 ? 1 : Factorial(number - 1) * number; // pass
}

TEST_CASE("Factorial of 0 is 1 (fail)", "[single-file]")
{
    REQUIRE(Factorial(0) == 1);
}

TEST_CASE("Factorials of 1 and higher are computed (pass)", "[single-file]")
{
    REQUIRE(Factorial(1) == 1);
    REQUIRE(Factorial(2) == 2);
    REQUIRE(Factorial(3) == 6);
    REQUIRE(Factorial(10) == 3628800);
}

TEST_CASE("Random threaded test", "[single-file]")
{
    dabun::thread::operating_cpu_set oset({0, 1, 2, 3, 4});
    // int                              i;
    // std::cin >> i;
    std::cout << "Was sleeping? "
              << (oset.set_sleeping_mode(true) ? " Yes" : "No") << std::endl;

    // std::cin >> i;

    for (int i = 0; i < 10000; ++i)
    {
        int x = rand() % 2;
        // std::cout << "Requesting: "
        //           << (oset.set_sleeping_mode(x) ? " Yes" : "No") << ' ';
        // std::cout << "Was sleeping? "
        //           << (oset.set_sleeping_mode(x) ? " Yes" : "No") <<
        //           std::endl;
        oset.set_sleeping_mode(x);
    }

    // std::cin >> i;
}
