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

#ifndef MAX_INTERPRETED_DEPTH
#define MAX_INTERPRETED_DEPTH 0
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
            {{"AcBr", BcCc}, {"BcCc", 1}}, 1, 322, nullptr, {}, nullptr, {},
            std::nullopt, MAX_INTERPRETED_DEPTH);

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
            {{"AcBr", BcCc}, {"BcCc", 1}}, 0, 322, nullptr, {}, nullptr, {},
            std::nullopt, MAX_INTERPRETED_DEPTH);

        auto fn = tree.get_fn();

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        // this time not accumulating
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);

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
            {{"R", 1}, {"C", C}}, std::nullopt, MAX_INTERPRETED_DEPTH);

        auto transpose = facebook::sysml::aot::transposer_baseline(
            {{"R", 1}, {"C", 1}}, {{"R", R}, {"C", C}}, {{"R", 1}, {"C", R}},
            {{"R", 1}, {"C", C}});

        transpose(B.data(), A.data());

        auto fn = tree.get_fn();
        fn({{"A", A.data()}, {"C", BJ.data()}});

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(BJ.data(), BJ.data() + R * C, B.data()) << "\n";
    }

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
            {{"AcBr", BcCc}, {"BcCc", 1}}, 1, 40, nullptr, {}, nullptr, {},
            std::nullopt, MAX_INTERPRETED_DEPTH);

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

#ifndef NELEMENTWISE
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
            {{"AcBr", BcCc}, {"BcCc", 1}}, 0, 1024, nullptr, {},
            compose(elementwise_bias<CT_ISA>, elementwise_relu<CT_ISA>),
            {
                {{"BcCc", 1}},
            },
            std::nullopt, MAX_INTERPRETED_DEPTH);

        auto fn = tree.get_fn();

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        auto bias = getRandomVector<float>(1 * BcCc * 1);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        apply_relu(CN.data(), CN.data() + ArCr * BcCc);

        fn({{"C", CJ.data()},
            {"A", A.data()},
            {"B", B.data()},
            {"post", bias.data()}});

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

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
            {{"AcBr", BcCc}, {"BcCc", 1}}, 1, 1024,
            compose(elementwise_bias<CT_ISA>, elementwise_relu<CT_ISA>),
            {
                {{"BcCc", 1}},
            },
            nullptr, {}, std::nullopt, MAX_INTERPRETED_DEPTH);

        auto fn = tree.get_fn();

        auto A = getRandomVector<float>(AcBr * ArCr);
        auto B = getRandomVector<float>(AcBr * BcCc);

        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        auto bias = getRandomVector<float>(1 * BcCc * 1);

        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        apply_relu(CN.data(), CN.data() + ArCr * BcCc);
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn({{"C", CJ.data()},
            {"A", A.data()},
            {"B", B.data()},
            {"pre", bias.data()}});

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }
#endif

    {
        /*
        for BcCc:
            for ArCr:
                // no zero-ing out of C1,C2
                for AcBr1:
                    C1 += A1 * B1
                for AcBr2:
                    C2 += A2 * B2
        */

        int ArCr  = 100;
        int AcBr1 = 100;
        int AcBr2 = 200;
        int BcCc  = 100;

        std::map<std::string, int> sizes = {
            {"ArCr", ArCr}, {"AcBr1", AcBr1}, {"AcBr2", AcBr2}, {"BcCc", BcCc}};

        std::map<std::string, std::set<std::string>> formulas = {
            {"C1", {"ArCr", "BcCc"}},  {"A1", {"ArCr", "AcBr1"}},
            {"B1", {"AcBr1", "BcCc"}}, {"C2", {"ArCr", "BcCc"}},
            {"A2", {"ArCr", "AcBr2"}}, {"B2", {"AcBr2", "BcCc"}}};

        std::map<std::string, std::map<std::string, int>> mm1_strides = {
            {"C1", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A1", {{"ArCr", AcBr1}, {"AcBr1", 1}}},
            {"B1", {{"AcBr1", BcCc}, {"BcCc", 1}}}};

        auto mm1 = make_compute_node<CT_ISA>(
            {"A1", "B1"}, "C1", mm1_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 100);

        std::map<std::string, std::map<std::string, int>> mm2_strides = {
            {"C2", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A2", {{"ArCr", AcBr2}, {"AcBr2", 1}}},
            {"B2", {{"AcBr2", BcCc}, {"BcCc", 1}}}};

        auto mm2 = make_compute_node<CT_ISA>(
            {"A2", "B2"}, "C2", mm2_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 100);

        auto ln1 = make_for_loop_node<CT_ISA>("AcBr1", 1, {mm1});
        auto ln2 = make_for_loop_node<CT_ISA>("AcBr2", 1, {mm2});

        auto root = make_for_loop_node<CT_ISA>(
            "BcCc", 1, {make_for_loop_node<CT_ISA>("ArCr", 1, {ln1, ln2})});

        auto tree = make_loop_tree_program<CT_ISA>({root}, sizes, formulas,
                                                   MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        auto A1  = getRandomVector<float>(AcBr1 * ArCr);
        auto B1  = getRandomVector<float>(AcBr1 * BcCc);
        auto CN1 = getRandomVector<float>(ArCr * BcCc);
        auto CJ1 = CN1;

        auto A2  = getRandomVector<float>(AcBr2 * ArCr);
        auto B2  = getRandomVector<float>(AcBr2 * BcCc);
        auto CN2 = getRandomVector<float>(ArCr * BcCc);
        auto CJ2 = CN2;

        baseline_MM(ArCr, AcBr1, BcCc, AcBr1, 1, BcCc, 1, BcCc, 1, A1.data(),
                    B1.data(), CN1.data(), 1);

        baseline_MM(ArCr, AcBr2, BcCc, AcBr2, 1, BcCc, 1, BcCc, 1, A2.data(),
                    B2.data(), CN2.data(), 1);

        std::map<std::string, float*> tensors = {
            {"C1", CJ1.data()}, {"A1", A1.data()}, {"B1", B1.data()},
            {"C2", CJ2.data()}, {"A2", A2.data()}, {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ1.data(), CJ1.data() + ArCr * BcCc,
                                CN1.data())
                  << "\n";

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ2.data(), CJ2.data() + ArCr * BcCc,
                                CN2.data())
                  << "\n";
    }

    {
        /*
        for BcCc:
            for ArCr:
                C1 = 0
                C2 = 0
                for AcBr1:
                    C1 += A1 * B1
                for AcBr2:
                    C2 += A2 * B2
        */

        int ArCr  = 100;
        int AcBr1 = 100;
        int AcBr2 = 200;
        int BcCc  = 100;

        std::map<std::string, int> sizes = {
            {"ArCr", ArCr}, {"AcBr1", AcBr1}, {"AcBr2", AcBr2}, {"BcCc", BcCc}};

        std::map<std::string, std::set<std::string>> formulas = {
            {"C1", {"ArCr", "BcCc"}},  {"A1", {"ArCr", "AcBr1"}},
            {"B1", {"AcBr1", "BcCc"}}, {"C2", {"ArCr", "BcCc"}},
            {"A2", {"ArCr", "AcBr2"}}, {"B2", {"AcBr2", "BcCc"}}};

        std::map<std::string, std::map<std::string, int>> mm1_strides = {
            {"C1", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A1", {{"ArCr", AcBr1}, {"AcBr1", 1}}},
            {"B1", {{"AcBr1", BcCc}, {"BcCc", 1}}}};

        auto mm1 = make_compute_node<CT_ISA>(
            {"A1", "B1"}, "C1", mm1_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        std::map<std::string, std::map<std::string, int>> mm2_strides = {
            {"C2", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A2", {{"ArCr", AcBr2}, {"AcBr2", 1}}},
            {"B2", {{"AcBr2", BcCc}, {"BcCc", 1}}}};

        auto mm2 = make_compute_node<CT_ISA>(
            {"A2", "B2"}, "C2", mm2_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        auto ln1 = make_for_loop_node<CT_ISA>("AcBr1", 1, {mm1});
        auto ln2 = make_for_loop_node<CT_ISA>("AcBr2", 1, {mm2});

        auto root = make_for_loop_node<CT_ISA>(
            "BcCc", 1, {make_for_loop_node<CT_ISA>("ArCr", 1, {ln1, ln2})});

        auto tree = make_loop_tree_program<CT_ISA>({root}, sizes, formulas,
                                                   MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        auto A1  = getRandomVector<float>(AcBr1 * ArCr);
        auto B1  = getRandomVector<float>(AcBr1 * BcCc);
        auto CN1 = getRandomVector<float>(ArCr * BcCc);
        auto CJ1 = CN1;

        auto A2  = getRandomVector<float>(AcBr2 * ArCr);
        auto B2  = getRandomVector<float>(AcBr2 * BcCc);
        auto CN2 = getRandomVector<float>(ArCr * BcCc);
        auto CJ2 = CN2;

        baseline_MM(ArCr, AcBr1, BcCc, AcBr1, 1, BcCc, 1, BcCc, 1, A1.data(),
                    B1.data(), CN1.data(), 0);

        baseline_MM(ArCr, AcBr2, BcCc, AcBr2, 1, BcCc, 1, BcCc, 1, A2.data(),
                    B2.data(), CN2.data(), 0);

        std::map<std::string, float*> tensors = {
            {"C1", CJ1.data()}, {"A1", A1.data()}, {"B1", B1.data()},
            {"C2", CJ2.data()}, {"A2", A2.data()}, {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ1.data(), CJ1.data() + ArCr * BcCc,
                                CN1.data())
                  << "\n";

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ2.data(), CJ2.data() + ArCr * BcCc,
                                CN2.data())
                  << "\n";
    }

    {
        /*
        for BcCc:
            for ArCr:
                // no zero-ing out of C1
                for AcBr1:
                    C1 += A1 * B1

        for BcCc:
            for ArCr:
                C2 = 0
                for AcBr2:
                    C2 += A2 * B2
        */

        int ArCr  = 100;
        int AcBr1 = 100;
        int AcBr2 = 200;
        int BcCc  = 100;

        std::map<std::string, int> sizes = {
            {"ArCr", ArCr}, {"AcBr1", AcBr1}, {"AcBr2", AcBr2}, {"BcCc", BcCc}};

        std::map<std::string, std::set<std::string>> formulas = {
            {"C1", {"ArCr", "BcCc"}},  {"A1", {"ArCr", "AcBr1"}},
            {"B1", {"AcBr1", "BcCc"}}, {"C2", {"ArCr", "BcCc"}},
            {"A2", {"ArCr", "AcBr2"}}, {"B2", {"AcBr2", "BcCc"}}};

        std::map<std::string, std::map<std::string, int>> mm1_strides = {
            {"C1", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A1", {{"ArCr", AcBr1}, {"AcBr1", 1}}},
            {"B1", {{"AcBr1", BcCc}, {"BcCc", 1}}}};

        auto mm1 = make_compute_node<CT_ISA>(
            {"A1", "B1"}, "C1", mm1_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 100);

        std::map<std::string, std::map<std::string, int>> mm2_strides = {
            {"C2", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A2", {{"ArCr", AcBr2}, {"AcBr2", 1}}},
            {"B2", {{"AcBr2", BcCc}, {"BcCc", 1}}}};

        auto mm2 = make_compute_node<CT_ISA>(
            {"A2", "B2"}, "C2", mm2_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        auto ln1 = make_for_loop_node<CT_ISA>(
            "BcCc", 1,
            {make_for_loop_node<CT_ISA>(
                "ArCr", 1, {make_for_loop_node<CT_ISA>("AcBr1", 1, {mm1})})});

        auto ln2 = make_for_loop_node<CT_ISA>(
            "BcCc", 1,
            {make_for_loop_node<CT_ISA>(
                "ArCr", 1, {make_for_loop_node<CT_ISA>("AcBr2", 1, {mm2})})});

        auto tree = make_loop_tree_program<CT_ISA>({ln1, ln2}, sizes, formulas,
                                                   MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        auto A1  = getRandomVector<float>(AcBr1 * ArCr);
        auto B1  = getRandomVector<float>(AcBr1 * BcCc);
        auto CN1 = getRandomVector<float>(ArCr * BcCc);
        auto CJ1 = CN1;

        auto A2  = getRandomVector<float>(AcBr2 * ArCr);
        auto B2  = getRandomVector<float>(AcBr2 * BcCc);
        auto CN2 = getRandomVector<float>(ArCr * BcCc);
        auto CJ2 = CN2;

        baseline_MM(ArCr, AcBr1, BcCc, AcBr1, 1, BcCc, 1, BcCc, 1, A1.data(),
                    B1.data(), CN1.data(), 1);

        baseline_MM(ArCr, AcBr2, BcCc, AcBr2, 1, BcCc, 1, BcCc, 1, A2.data(),
                    B2.data(), CN2.data(), 0);

        std::map<std::string, float*> tensors = {
            {"C1", CJ1.data()}, {"A1", A1.data()}, {"B1", B1.data()},
            {"C2", CJ2.data()}, {"A2", A2.data()}, {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ1.data(), CJ1.data() + ArCr * BcCc,
                                CN1.data())
                  << "\n";

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ2.data(), CJ2.data() + ArCr * BcCc,
                                CN2.data())
                  << "\n";
    }

    {
        /*
        for BcCc:
            for ArCr1:
                C1 = 0
                for AcBr1:
                    C1 += A1 * B1
            for ArCr2:
                // no zero-ing out of C2
                for AcBr2:
                    C2 += A2 * B2
        */

        int ArCr1 = 100;
        int ArCr2 = 50;
        int AcBr1 = 100;
        int AcBr2 = 200;
        int BcCc  = 100;

        std::map<std::string, int> sizes = {{"ArCr1", ArCr1},
                                            {"ArCr2", ArCr2},
                                            {"AcBr1", AcBr1},
                                            {"AcBr2", AcBr2},
                                            {"BcCc", BcCc}};

        std::map<std::string, std::set<std::string>> formulas = {
            {"C1", {"ArCr1", "BcCc"}},  {"A1", {"ArCr1", "AcBr1"}},
            {"B1", {"AcBr1", "BcCc"}},  {"C2", {"ArCr2", "BcCc"}},
            {"A2", {"ArCr2", "AcBr2"}}, {"B2", {"AcBr2", "BcCc"}}};

        std::map<std::string, std::map<std::string, int>> mm1_strides = {
            {"C1", {{"ArCr1", BcCc}, {"BcCc", 1}}},
            {"A1", {{"ArCr1", AcBr1}, {"AcBr1", 1}}},
            {"B1", {{"AcBr1", BcCc}, {"BcCc", 1}}}};

        auto mm1 = make_compute_node<CT_ISA>(
            {"A1", "B1"}, "C1", mm1_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        std::map<std::string, std::map<std::string, int>> mm2_strides = {
            {"C2", {{"ArCr2", BcCc}, {"BcCc", 1}}},
            {"A2", {{"ArCr2", AcBr2}, {"AcBr2", 1}}},
            {"B2", {{"AcBr2", BcCc}, {"BcCc", 1}}}};

        auto mm2 = make_compute_node<CT_ISA>(
            {"A2", "B2"}, "C2", mm2_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 100);

        auto ln1 = make_for_loop_node<CT_ISA>(
            "ArCr1", 1, {make_for_loop_node<CT_ISA>("AcBr1", 1, {mm1})});

        auto ln2 = make_for_loop_node<CT_ISA>(
            "ArCr2", 1, {make_for_loop_node<CT_ISA>("AcBr2", 1, {mm2})});

        auto root = make_for_loop_node<CT_ISA>("BcCc", 1, {ln1, ln2});

        auto tree = make_loop_tree_program<CT_ISA>({root}, sizes, formulas,
                                                   MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        auto A1  = getRandomVector<float>(AcBr1 * ArCr1);
        auto B1  = getRandomVector<float>(AcBr1 * BcCc);
        auto CN1 = getRandomVector<float>(ArCr1 * BcCc);
        auto CJ1 = CN1;

        auto A2  = getRandomVector<float>(AcBr2 * ArCr2);
        auto B2  = getRandomVector<float>(AcBr2 * BcCc);
        auto CN2 = getRandomVector<float>(ArCr2 * BcCc);
        auto CJ2 = CN2;

        baseline_MM(ArCr1, AcBr1, BcCc, AcBr1, 1, BcCc, 1, BcCc, 1, A1.data(),
                    B1.data(), CN1.data(), 0);

        baseline_MM(ArCr2, AcBr2, BcCc, AcBr2, 1, BcCc, 1, BcCc, 1, A2.data(),
                    B2.data(), CN2.data(), 1);

        std::map<std::string, float*> tensors = {
            {"C1", CJ1.data()}, {"A1", A1.data()}, {"B1", B1.data()},
            {"C2", CJ2.data()}, {"A2", A2.data()}, {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ1.data(), CJ1.data() + ArCr1 * BcCc,
                                CN1.data())
                  << "\n";

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ2.data(), CJ2.data() + ArCr2 * BcCc,
                                CN2.data())
                  << "\n";
    }

    {
        /*
        C = 0
        for BcCc:
            for AcBr:
                // transpose B1 into B2
                // (interpreted, since parent can't be jitted)
                B2[BcCc, AcBr] = B1[AcBr, BcCc]
                for ArCr:
                    // jitted
                    C[ArCr, BcCc] += A[ArCr, AcBr] * B2[BcCc, AcBr]
        */

        int ArCr = 100;
        int AcBr = 100;
        int BcCc = 100;

        std::map<std::string, int> sizes = {
            {"ArCr", ArCr}, {"AcBr", AcBr}, {"BcCc", BcCc}};

        std::map<std::string, std::set<std::string>> formulas = {
            {"C", {"ArCr", "BcCc"}},
            {"A", {"ArCr", "AcBr"}},
            {"B1", {"AcBr", "BcCc"}},
            {"B2", {"AcBr", "BcCc"}}};

        std::map<std::string, std::map<std::string, int>> mm_strides = {
            {"C", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A", {{"ArCr", AcBr}, {"AcBr", 1}}},
            {"B2", {{"AcBr", 1}, {"BcCc", AcBr}}}};

        auto mm = make_compute_node<CT_ISA>(
            {"A", "B2"}, "C", mm_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        std::map<std::string, std::map<std::string, int>> transpose_strides = {
            {"B1", {{"AcBr", BcCc}, {"BcCc", 1}}},
            {"B2", {{"AcBr", 1}, {"BcCc", AcBr}}}};

        auto tr =
            make_transpose_node<CT_ISA>("B1", "B2", transpose_strides, 100);

        auto root = make_for_loop_node<CT_ISA>(
            "BcCc", 1,
            {make_for_loop_node<CT_ISA>(
                "AcBr", 1, {tr, make_for_loop_node<CT_ISA>("ArCr", 1, {mm})})});

        auto tree = make_loop_tree_program<CT_ISA>({root}, sizes, formulas,
                                                   MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        auto A  = getRandomVector<float>(AcBr * ArCr);
        auto B1 = getRandomVector<float>(AcBr * BcCc);
        auto B2 = getRandomVector<float>(AcBr * BcCc);
        auto CN = getRandomVector<float>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B1.data(), CN.data(), 0);

        std::map<std::string, float*> tensors = {{"C", CJ.data()},
                                                 {"A", A.data()},
                                                 {"B1", B1.data()},
                                                 {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + ArCr * BcCc, CN.data())
                  << "\n";
    }

    {

        int CIN  = 128;
        int COUT = 128 + 3;
        int OS   = 56 + 4;
        int KS   = 3;
        int IS   = OS + KS - 1;

        auto tree = loop_tree_program<CT_ISA>(
            {{"c_out", 16}, //
             {"o_h", 1},
             {"o_w", 28},
             {"c_in", 16},
             {"c_in", 1},
             {"o_w", 1}, //
             //{"o_w", 1},    //
             {"k_h", 1},    //
             {"k_w", 1},    //
             {"c_out", 1}}, //
            // The second argument is a map of the dimension sizes
            {{"c_out", COUT},
             {"o_w", OS},
             {"k_w", KS},
             {"c_in", CIN},
             {"o_h", OS},
             {"k_h", KS}},
            // Vars of C (other variables are reduction variables)
            {"c_out", "o_w", "o_h"},
            // Variables of A, note that i_w and i_h are not used
            {"c_in", "i_w", "i_h"},
            // Variables of B
            {"c_in", "c_out", "k_w", "k_h"},
            // C's strides for each variable
            {{"o_w", COUT}, {"c_out", 1}, {"o_h", COUT * OS}},
            // A's strides for each variable Note how we
            // provide strides for i/k_h and i/k_w, this is
            // because the access to A is based on output
            // and reduction variables
            {{"o_w", CIN},
             {"k_w", CIN},
             {"c_in", 1},
             {"o_h", IS * CIN},
             {"k_h", IS * CIN}},
            // B's strides for each variable
            {{"c_out", 1},
             {"c_in", COUT},
             {"k_w", COUT * CIN},
             {"k_h", COUT * CIN * KS}},
            0, 250, nullptr, {}, nullptr, {}, std::nullopt,
            MAX_INTERPRETED_DEPTH);

        auto fn = tree.get_fn();

        auto A  = getRandomVector<float>(CIN * IS * IS);
        auto B  = getRandomVector<float>(COUT * CIN * KS * KS);
        auto CN = std::vector<float>(COUT * OS * OS);
        auto CJ = std::vector<float>(COUT * OS * OS);

        baseline_Conv(COUT, CIN, OS, OS, KS, KS, A.data(), B.data(), CN.data());

        fn({{"C", CJ.data()}, {"A", A.data()}, {"B", B.data()}});

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + COUT * OS * OS,
                                CN.data())
                  << "\n";
    }

    {
        int OX = 101;
        int OY = 101;
        int OZ = 16 * 12 + 3;
        int KX = 3;
        int KY = 3;
        int KZ = 3;
        int IX = OX + KX - 1;
        int IY = OY + KY - 1;
        int IZ = OZ + KZ - 1;

        auto tree = loop_tree_program<CT_ISA>(
            // The first argument is the loop order in the form of
            // {dimension, stride}.  For now the outer dimension has
            // to divide the stride.  This is effectively the same as
            // Halide's split into outer and inner variable, but can
            // have arbitray number of splits.
            {{"OX", 1},  // To block B in L2 cache
             {"OY", 10}, // This and the next are for the register
                         // blocking of C - 30 vector registers of
                         // each holding 16 values
             {"OY", 1},
             {"OZ", 16},
             {"KX", 1}, // broken up to allow for unrolling of 4
             {"KY", 1}, // inner loops, should handle differently
                        // later
             {"KZ", 1},
             {"OZ", 1}},
            // The second argument is a map of the dimension sizes
            {{"OX", OX},
             {"OY", OY},
             {"OZ", OZ},
             {"KX", KX},
             {"KY", KY},
             {"KZ", KZ}},
            // Vars of C (other variables are reduction variables)
            {"OX", "OY", "OZ"},
            // Variables of A
            {"IX", "IY", "IZ"},
            // Variables of B
            {"KX", "KY", "KZ"},
            // C's strides for each variable.  Note that the strides
            // data is a superset of the previous argument (variables
            // of C).  I'm still deciding on the final design,
            // possibly allowing for null strides that will just
            // deduce them from the sizes, or some special structs
            // indicating the layout (ie row-major, col-major).  In
            // this case the vars have to be ordered though...
            // Many decisions to make...
            {{"OX", OY * OZ}, {"OY", OZ}, {"OZ", 1}},
            // A's strides for each variable
            {{"OX", IY * IZ},
             {"OY", IZ},
             {"OZ", 1},
             {"KX", IY * IZ},
             {"KY", IZ},
             {"KZ", 1}},
            // B's strides for each variable
            {{"KX", KY * KZ}, {"KY", KZ}, {"KZ", 1}}, 0, 1024, nullptr, {},
            facebook::sysml::aot::elementwise_relu<CT_ISA>, {}, std::nullopt,
            MAX_INTERPRETED_DEPTH);

        auto fn = tree.get_fn();
        auto A  = getRandomVector<float>(IX * IY * IZ);
        auto B  = getRandomVector<float>(KX * KY * KZ);

        auto CN = std::vector<float>(OX * OY * OZ);
        auto CJ = std::vector<float>(OX * OY * OZ);

        baseline_3DConv(OX, OY, OZ, KX, KY, KZ, A.data(), B.data(), CN.data());
        apply_relu(CN.data(), CN.data() + CN.size());

        fn({{"C", CJ.data()}, {"A", A.data()}, {"B", B.data()}});

        std::cout << "MAXABSDIFF: "
                  << maxAbsDiff(CJ.data(), CJ.data() + OX * OY * OZ, CN.data())
                  << "\n";
    }
}
