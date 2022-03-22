#include "dabun/thread/cpu_pool.hpp"
#include "dabun/thread/parallel_for.hpp"
#include "dabun/utility/for_all.hpp"
#include "dabun/utility/vek.hpp"

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
    // dabun::thread::cpu_pool oset({0, 1, 5, 12, 18});
    dabun::thread::cpu_pool oset(10);
    //  int                              i;
    //  std::cin >> i;
    //  std::cout << "Was sleeping? "
    //            << (oset.set_sleeping_mode(true) ? " Yes" : "No") <<
    //            std::endl;

    // std::cin >> i;

    // for (int i = 0; i < 10000000; ++i)
    // {
    //     int x = rand() % 2;
    //     // std::cout << "Requesting: "
    //     //           << (oset.set_sleeping_mode(x) ? " Yes" : "No") << ' ';
    //     // std::cout << "Was sleeping? "
    //     //           << (oset.set_sleeping_mode(x) ? " Yes" : "No") <<
    //     //           std::endl;
    //     oset.set_sleeping_mode(x);
    // }

    int const len = 10000;

    {
        std::vector<int> all_zeros(len);

        dabun::thread::naive_parallel_for(oset, 0, len, 1,
                                          [&](int idx) { all_zeros[idx] = 1; });

        REQUIRE(std::accumulate(all_zeros.begin(), all_zeros.end(), 0) == len);
    }

    {
        std::vector<int> all_zeros(len);

        dabun::thread::single_queue_parallel_for(oset, 0, len, 1,
                                                 [&](auto const& c, int idx)
                                                 {
                                                     std::cout << c.cpu_index
                                                               << "\n";
                                                     all_zeros[idx] = 1;
                                                 });

        REQUIRE(std::accumulate(all_zeros.begin(), all_zeros.end(), 0) == len);
    }

    std::atomic<int> zi{0};

    oset.execute_on_all_cpus(
        [&](auto)
        {
            while (zi.fetch_add(1) < len)
            {
            }
        });

    REQUIRE(zi.load() == len + oset.size());

    std::cout
        << std::alignment_of_v<
               dabun::detail::primitive_aligned_wrapper<int, 1024>> << "\n\n";

    std::cout << sizeof(dabun::detail::primitive_aligned_wrapper<int, 1024>)
              << "\n\n";

    // int i;
    // std::cin >> i;
    {
        // using dabun::vek;

        // vek<int, 1> b1 = std::array<int, 1>{1};
        // vek<int, 2> b2{{1, 1}};

        // auto begi = concat(b1, b2);

        // auto begin = begi + 1;

        // // dabun::vek<int, 3> begin{1, 1, 1};
        // dabun::vek<int, 3> end{{3, 4, 5}};
        // end += 1;

        // std::cout << (b2 == b2) << " " << (b2 != b2) << "\n";

        // dabun::coord_for_loop(begin, end,
        //                       [](auto const& v)
        //                       {
        //                           std::cout << v << "\n";
        //                       });

        // std::cout << "\n";
        // std::cout << "\n";
        // std::cout << "\n";
        // std::cout << "\n";

        // // dabun::coord_for_loop(end,
        // //                       [](auto const& v)
        // //                       {
        // //                           std::cout << v[0];
        // //                           for (int i = 1; i < v.size(); ++i)
        // //                           {
        // //                               std::cout << ", " << v[i];
        // //                           }
        // //                           std::cout << "\n";
        // //                       });

        // std::cout << -(end + 3) << "\n";


        // std::cout << dabun::to_string(-end + 3, ',') << "\n";

        // {
        //     auto r = -end + 3;
        //     // auto z = dabun::head<2>(r);
        //     std::cout << dabun::head<2>(r) << "\n";
        //     // std::cout << z << "\n";

        // }



    }
}
