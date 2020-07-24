// Copyright 2004-present Facebook. All Rights Reserved.

#include <iostream>

#include "arithmetic_operation.h"
#include "baselines.h"
#include "loop_tree.h"
#include "transposer.h"
#include "transposer_baseline.h"
#include "transposer_bench.h"
#include "utils.h"

#ifndef CT_ISA
#define CT_ISA avx2
#endif

using namespace facebook::sysml::aot;

int main()
{
    {
        int ArCr = 100;
        int AcBr = 100;
        int BcCc = 100;

        auto tree = loop_tree_program<CT_ISA>(
            {{"AcBr", 256},
             {"ArCr", 3},
             {"BcCc", 16},
             {"AcBr", 1},
             {"AcBr", 1},
             {"ArCr", 1},
             {"BcCc", 1}},
            // The second argument is a map of the dimension sizes
            {{"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}},
            // Vars of C (other variables are reduction variables)
            {"ArCr", "BcCc"},
            // Variables of A
            {"ArCr", "AcBr"},
            // Variables of B
            {"AcBr", "BcCc"},
            // C's strides for each variable.  Note that the
            // strides data is a superset of the previous argument
            // (variables of C).  I'm still deciding on the final
            // design, possibly allowing for null strides that
            // will just deduce them from the sizes, or some
            // special structs indicating the layout (ie
            // row-major, col-major).  In this case the vars have
            // to be ordered though... Many decisions to make...
            {{"ArCr", BcCc}, {"BcCc", 1}},
            // A's strides for each variable
            {{"ArCr", AcBr}, {"AcBr", 1}},
            // B's strides for each variable
            {{"AcBr", BcCc}, {"BcCc", 1}});

        auto fn = tree.get_fn();

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn({{"C", CJ.data()}, {"A", A.data()}, {"B", B.data()}});

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {

        int R = 1024;
        int C = 1024;

        auto A  = getRandomVector<float>(R * C);
        auto B  = getRandomVector<float>(R * C);
        auto BJ = getRandomVector<float>(R * C);

        auto tree = loop_tree_program<CT_ISA>(
            {{"R", 1}, {"C", 1}}, {{"R", R}, {"C", C}}, {{"R", 1}, {"C", R}},
            {{"R", 1}, {"C", C}});


        auto transpose = facebook::sysml::aot::transposer_baseline(
            {{"R", 1}, {"C", 1}}, {{"R", R}, {"C", C}}, {{"R", 1}, {"C", R}},
            {{"R", 1}, {"C", C}});

        transpose(B.data(), A.data());

        auto fn = tree.get_fn();
        fn({{"A", A.data()}, {"C", BJ.data()}});

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(BJ.data(), BJ.data() + R * C, B.data()) << "\n";
    }
}
