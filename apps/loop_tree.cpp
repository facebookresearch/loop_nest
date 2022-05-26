// Copyright 2004-present Facebook. All Rights Reserved.

#include <iostream>

#include "baselines.hpp"
#include "dabun/arithmetic_operation.hpp"
#include "dabun/check.hpp"
#include "dabun/loop_tree/program.hpp"
#include "dabun/numeric.hpp"
#include "dabun/random_vector.hpp"
#include "dabun/transposer.hpp"
#include "transposer_baseline.hpp"
#include "transposer_bench.hpp"
#include "utility.hpp"

#ifndef DABUN_ISA
#    define DABUN_ISA avx2
#endif

// just for testing
// some examples are expensive to do
// when testing no jitter
#ifndef SKIP_EXPENSIVE
#    define SKIP_EXPENSIVE false
#endif

// just for testing:
// forces a prefix in loop tree
// to be interpreted (rather than
// part of jitted loop nest)
#ifndef MAX_INTERPRETED_DEPTH
#    define MAX_INTERPRETED_DEPTH 0
#endif

// TODO(zi) relax this when ARM implementation gets elementwise(bias) support
#if defined(__aarch64__) && !defined(NELEMENTWISE)
#    define NELEMENTWISE
#endif

using namespace dabun;
using namespace dabun::loop_tree;

int main()
{

    using float_t = DABUN_ARITHMETIC;

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

        /*
       ----------> Definition
       */

        int ArCr = 128;
        int AcBr = 128;
        int BcCc = 128;

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

        // C[ArCr, BcCc] += A[ArCr, AcBr] * B2[BcCc, AcBr] (with zero init)
        auto mm = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A", "B2"}, "C", mm_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100, nullptr, {},
            elementwise_relu<DABUN_ISA>);

        /*
        for ArCr:
            // jitted
            C[ArCr, BcCc] += A[ArCr, AcBr] * B2[BcCc, AcBr]
        */
        auto ln =
            make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("ArCr", 1, {mm});

        std::map<std::string, std::map<std::string, int>> transpose_strides = {
            {"B1", {{"AcBr", BcCc}, {"BcCc", 1}}},
            {"B2", {{"AcBr", 1}, {"BcCc", AcBr}}}};

        // B2[BcCc, AcBr] = B1[AcBr, BcCc]
        auto tr = make_transpose_node<DABUN_VEX, DABUN_ARITHMETIC>(
            "B1", "B2", transpose_strides, 100);

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
        auto root = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
            "BcCc", 1,
            {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr", 1,
                                                             {tr, ln})},
            true);

        auto tree = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            {root}, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        std::cout << print_report(fn.get_report());

        /*
       ----------> Execution
       */

        auto A  = get_random_vector<float_t>(AcBr * ArCr);
        auto B1 = get_random_vector<float_t>(AcBr * BcCc);
        auto B2 = get_random_vector<float_t>(AcBr * BcCc);
        auto CN = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B1.data(), CN.data(), 0);
        apply_relu(CN.data(), CN.data() + ArCr * BcCc);

        std::map<std::string, float_t*> tensors = {{"C", CJ.data()},
                                                   {"A", A.data()},
                                                   {"B1", B1.data()},
                                                   {"B2", B2.data()}};

        // while (1)
        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";
    }

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

        /*
       ----------> Definition
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

        // C1 += A1 * B1
        // alpha 1 -> accumulate (no zero init)
        auto mm1 = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A1", "B1"}, "C1", mm1_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 100);

        /*
        for AcBr1:
            C1 += A1 * B1
        */
        auto ln1 =
            make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr1", 1, {mm1});

        std::map<std::string, std::map<std::string, int>> mm2_strides = {
            {"C2", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A2", {{"ArCr", AcBr2}, {"AcBr2", 1}}},
            {"B2", {{"AcBr2", BcCc}, {"BcCc", 1}}}};

        // C2 += A2 * B2
        auto mm2 = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A2", "B2"}, "C2", mm2_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 100);

        /*
        for AcBr2:
            C2 += A2 * B2
        */
        auto ln2 =
            make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr2", 1, {mm2});

        /*
        for BcCc:
            for ArCr:
                // no zero-ing out of C1, C2
                for AcBr1:
                    C1 += A1 * B1
                for AcBr2:
                    C2 += A2 * B2
        */
        auto root = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
            "BcCc", 1,
            {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("ArCr", 1,
                                                             {ln1, ln2})});

        auto tree = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            {root}, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
        ----------> Execution
        */

        auto A1  = get_random_vector<float_t>(AcBr1 * ArCr);
        auto B1  = get_random_vector<float_t>(AcBr1 * BcCc);
        auto CN1 = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ1 = CN1;

        auto A2  = get_random_vector<float_t>(AcBr2 * ArCr);
        auto B2  = get_random_vector<float_t>(AcBr2 * BcCc);
        auto CN2 = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ2 = CN2;

        baseline_MM(ArCr, AcBr1, BcCc, AcBr1, 1, BcCc, 1, BcCc, 1, A1.data(),
                    B1.data(), CN1.data(), 1);

        baseline_MM(ArCr, AcBr2, BcCc, AcBr2, 1, BcCc, 1, BcCc, 1, A2.data(),
                    B2.data(), CN2.data(), 1);

        std::map<std::string, float_t*> tensors = {
            {"C1", CJ1.data()}, {"A1", A1.data()}, {"B1", B1.data()},
            {"C2", CJ2.data()}, {"A2", A2.data()}, {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ1.data(), CJ1.data() + ArCr * BcCc,
                                        CN1.data())
                  << "\n";

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ2.data(), CJ2.data() + ArCr * BcCc,
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

        /*
        ----------> Definition
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

        // C1 += A1 * B1
        auto mm1 = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A1", "B1"}, "C1", mm1_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        /*
        for AcBr1:
            C1 += A1 * B1
        */
        auto ln1 =
            make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr1", 1, {mm1});

        std::map<std::string, std::map<std::string, int>> mm2_strides = {
            {"C2", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A2", {{"ArCr", AcBr2}, {"AcBr2", 1}}},
            {"B2", {{"AcBr2", BcCc}, {"BcCc", 1}}}};

        // C2 += A2 * B2
        auto mm2 = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A2", "B2"}, "C2", mm2_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        /*
        for AcBr2:
            C2 += A2 * B2
        */
        auto ln2 =
            make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr2", 1, {mm2});

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
        auto root = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
            "BcCc", 1,
            {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("ArCr", 1,
                                                             {ln1, ln2})});

        auto tree = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            {root}, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
        ----------> Execution
        */

        auto A1  = get_random_vector<float_t>(AcBr1 * ArCr);
        auto B1  = get_random_vector<float_t>(AcBr1 * BcCc);
        auto CN1 = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ1 = CN1;

        auto A2  = get_random_vector<float_t>(AcBr2 * ArCr);
        auto B2  = get_random_vector<float_t>(AcBr2 * BcCc);
        auto CN2 = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ2 = CN2;

        baseline_MM(ArCr, AcBr1, BcCc, AcBr1, 1, BcCc, 1, BcCc, 1, A1.data(),
                    B1.data(), CN1.data(), 0);

        baseline_MM(ArCr, AcBr2, BcCc, AcBr2, 1, BcCc, 1, BcCc, 1, A2.data(),
                    B2.data(), CN2.data(), 0);

        std::map<std::string, float_t*> tensors = {
            {"C1", CJ1.data()}, {"A1", A1.data()}, {"B1", B1.data()},
            {"C2", CJ2.data()}, {"A2", A2.data()}, {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ1.data(), CJ1.data() + ArCr * BcCc,
                                        CN1.data())
                  << "\n";

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ2.data(), CJ2.data() + ArCr * BcCc,
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

        /*
       ----------> Definition
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

        // C1 += A1 * B1
        auto mm1 = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A1", "B1"}, "C1", mm1_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 100);

        /*
        for BcCc:
            for ArCr:
                // no zero-ing out of C1
                for AcBr1:
                    C1 += A1 * B1
        */
        auto ln1 = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
            "BcCc", 1,
            {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                "ArCr", 1,
                {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr1", 1,
                                                                 {mm1})})});

        std::map<std::string, std::map<std::string, int>> mm2_strides = {
            {"C2", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A2", {{"ArCr", AcBr2}, {"AcBr2", 1}}},
            {"B2", {{"AcBr2", BcCc}, {"BcCc", 1}}}};

        // C2 += A2 * B2
        auto mm2 = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A2", "B2"}, "C2", mm2_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        /*
        for BcCc:
            for ArCr:
                C2 = 0
                for AcBr2:
                    C2 += A2 * B2
        */
        auto ln2 = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
            "BcCc", 1,
            {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                "ArCr", 1,
                {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr2", 1,
                                                                 {mm2})})});

        auto tree = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            {ln1, ln2}, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
       ----------> Execution
       */

        auto A1  = get_random_vector<float_t>(AcBr1 * ArCr);
        auto B1  = get_random_vector<float_t>(AcBr1 * BcCc);
        auto CN1 = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ1 = CN1;

        auto A2  = get_random_vector<float_t>(AcBr2 * ArCr);
        auto B2  = get_random_vector<float_t>(AcBr2 * BcCc);
        auto CN2 = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ2 = CN2;

        baseline_MM(ArCr, AcBr1, BcCc, AcBr1, 1, BcCc, 1, BcCc, 1, A1.data(),
                    B1.data(), CN1.data(), 1);

        baseline_MM(ArCr, AcBr2, BcCc, AcBr2, 1, BcCc, 1, BcCc, 1, A2.data(),
                    B2.data(), CN2.data(), 0);

        std::map<std::string, float_t*> tensors = {
            {"C1", CJ1.data()}, {"A1", A1.data()}, {"B1", B1.data()},
            {"C2", CJ2.data()}, {"A2", A2.data()}, {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ1.data(), CJ1.data() + ArCr * BcCc,
                                        CN1.data())
                  << "\n";

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ2.data(), CJ2.data() + ArCr * BcCc,
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

        /*
       ----------> Definition
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

        // C1 += A1 * B1 with zero init
        auto mm1 = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A1", "B1"}, "C1", mm1_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 100);

        /*
        for ArCr1:
            C1 = 0
            for AcBr1:
                C1 += A1 * B1
        */
        auto ln1 = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
            "ArCr1", 1,
            {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr1", 1,
                                                             {mm1})});

        std::map<std::string, std::map<std::string, int>> mm2_strides = {
            {"C2", {{"ArCr2", BcCc}, {"BcCc", 1}}},
            {"A2", {{"ArCr2", AcBr2}, {"AcBr2", 1}}},
            {"B2", {{"AcBr2", BcCc}, {"BcCc", 1}}}};

        // C2  += A2 * B2
        auto mm2 = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A2", "B2"}, "C2", mm2_strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 100);

        /*
        for ArCr2:
            // no zero-ing out of C2
            for AcBr2:
                C2 += A2 * B2
        */
        auto ln2 = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
            "ArCr2", 1,
            {make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("AcBr2", 1,
                                                             {mm2})});

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
        auto root = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>("BcCc", 1,
                                                                    {ln1, ln2});

        auto tree = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            {root}, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
       ----------> Execution
       */

        auto A1  = get_random_vector<float_t>(AcBr1 * ArCr1);
        auto B1  = get_random_vector<float_t>(AcBr1 * BcCc);
        auto CN1 = get_random_vector<float_t>(ArCr1 * BcCc);
        auto CJ1 = CN1;

        auto A2  = get_random_vector<float_t>(AcBr2 * ArCr2);
        auto B2  = get_random_vector<float_t>(AcBr2 * BcCc);
        auto CN2 = get_random_vector<float_t>(ArCr2 * BcCc);
        auto CJ2 = CN2;

        baseline_MM(ArCr1, AcBr1, BcCc, AcBr1, 1, BcCc, 1, BcCc, 1, A1.data(),
                    B1.data(), CN1.data(), 0);

        baseline_MM(ArCr2, AcBr2, BcCc, AcBr2, 1, BcCc, 1, BcCc, 1, A2.data(),
                    B2.data(), CN2.data(), 1);

        std::map<std::string, float_t*> tensors = {
            {"C1", CJ1.data()}, {"A1", A1.data()}, {"B1", B1.data()},
            {"C2", CJ2.data()}, {"A2", A2.data()}, {"B2", B2.data()}};

        fn(tensors);

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ1.data(), CJ1.data() + ArCr1 * BcCc,
                                        CN1.data())
                  << "\n";

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ2.data(), CJ2.data() + ArCr2 * BcCc,
                                        CN2.data())
                  << "\n";
    }

    // std::cout << "DIFF: LALA\n";

    {
        // Matrix multiplication without zero init

        /*
       ----------> Definition
       */
        int ArCr = 100;
        int AcBr = 100;
        int BcCc = 100;

        // schedule
        std::vector<std::pair<std::string, int>> order = {
            {"AcBr", 256}, {"ArCr", 3}, {"BcCc", 16}, {"AcBr", 1},
            {"AcBr", 1},   {"ArCr", 1}, {"BcCc", 1}};

        // sizes of dimensions
        std::map<std::string, int> sizes = {
            {"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}};

        // dimensions in each tensor
        std::map<std::string, std::set<std::string>> formulas = {
            {"C", {"ArCr", "BcCc"}},
            {"A", {"ArCr", "AcBr"}},
            {"B", {"AcBr", "BcCc"}}};

        // strides defining innermost computation
        std::map<std::string, std::map<std::string, int>> strides = {
            {"C", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A", {{"ArCr", AcBr}, {"AcBr", 1}}},
            {"B", {{"AcBr", BcCc}, {"BcCc", 1}}}};

        // C += A * B
        auto mm = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A", "B"}, "C", strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, 322);

        // add for-loops from the order
        auto curr = mm;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            curr = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                it->first, it->second, {curr});
        }

        auto nodes = {curr};
        auto tree  = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            nodes, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        auto A = get_random_vector<float_t>(AcBr * ArCr);
        auto B = get_random_vector<float_t>(AcBr * BcCc);

        auto CN = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn({{"C", CJ.data()}, {"A", A.data()}, {"B", B.data()}});

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";
    }

    {
        // Matrix multiplication with zero init
        /*
       ----------> Definition
       */
        int ArCr = 100;
        int AcBr = 100;
        int BcCc = 100;

        // schedule
        std::vector<std::pair<std::string, int>> order = {
            {"AcBr", 256}, {"ArCr", 3}, {"BcCc", 16}, {"AcBr", 1},
            {"AcBr", 1},   {"ArCr", 1}, {"BcCc", 1}};

        // sizes of dimensions
        std::map<std::string, int> sizes = {
            {"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}};

        // dimensions in each tensor
        std::map<std::string, std::set<std::string>> formulas = {
            {"C", {"ArCr", "BcCc"}},
            {"A", {"ArCr", "AcBr"}},
            {"B", {"AcBr", "BcCc"}}};

        // strides defining innermost computation
        std::map<std::string, std::map<std::string, int>> strides = {
            {"C", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A", {{"ArCr", AcBr}, {"AcBr", 1}}},
            {"B", {{"AcBr", BcCc}, {"BcCc", 1}}}};

        // C += A * B (with zero-init)
        auto mm = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A", "B"}, "C", strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, 322);

        // add for-loops from the order
        auto curr = mm;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            curr = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                it->first, it->second, {curr});
        }

        auto nodes = {curr};
        auto tree  = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            nodes, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        auto A = get_random_vector<float_t>(AcBr * ArCr);
        auto B = get_random_vector<float_t>(AcBr * BcCc);

        auto CN = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ = CN;

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);

        fn({{"C", CJ.data()}, {"A", A.data()}, {"B", B.data()}});

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";
    }

    {
        // transposing
        /*
       ----------> Definition
       */
        int R = 1024;
        int C = 1024;

        std::map<std::string, int> sizes = {{"R", R}, {"C", C}};
        std::map<std::string, std::set<std::string>> formulas = {
            {"A", {"R", "C"}}, {"C", {"R", "C"}}};

        std::vector<std::pair<std::string, int>> order = {{"R", 1}, {"C", 1}};

        std::map<std::string, int> out_strides = {{"R", 1}, {"C", R}};
        std::map<std::string, int> in_strides  = {{"R", 1}, {"C", C}};
        std::map<std::string, std::map<std::string, int>> strides = {
            {"A", in_strides}, {"C", out_strides}};

        // transpose "A" into "C"
        auto tr =
            make_transpose_node<DABUN_VEX, DABUN_ARITHMETIC>("A", "C", strides);

        // add for-loops from the order
        auto curr = tr;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            curr = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                it->first, it->second, {curr});
        }

        auto nodes = {curr};
        // technically formulas aren't used in transpose so not
        //  necessary here
        // (but are always a required parameter for now)
        auto tree = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            nodes, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
       ----------> Execution
       */
        auto A  = get_random_vector<float_t>(R * C);
        auto B  = get_random_vector<float_t>(R * C);
        auto BJ = get_random_vector<float_t>(R * C);

        auto transpose =
            transposer_baseline<float_t>(order, sizes, out_strides, in_strides);

        transpose(B.data(), A.data());

        fn({{"A", A.data()}, {"C", BJ.data()}});

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(BJ.data(), BJ.data() + R * C, B.data())
                  << "\n";
    }

#ifndef NELEMENTWISE
    {
        // Matrix multiplication with zero init
        // and bias followed by relu as post-op

        /*
        ----------> Definition
        */
        int ArCr = 100;
        int AcBr = 100;
        int BcCc = 100;

        // schedule
        std::vector<std::pair<std::string, int>> order = {
            {"AcBr", 256}, {"ArCr", 3}, {"BcCc", 16}, {"AcBr", 1},
            {"AcBr", 1},   {"ArCr", 1}, {"BcCc", 1}};

        // sizes of dimensions
        std::map<std::string, int> sizes = {
            {"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}};

        // dimensions in each tensor
        std::map<std::string, std::set<std::string>> formulas = {
            {"C", {"ArCr", "BcCc"}},
            {"A", {"ArCr", "AcBr"}},
            {"B", {"AcBr", "BcCc"}}};

        // strides defining innermost computation
        std::map<std::string, std::map<std::string, int>> strides = {
            {"C", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A", {{"ArCr", AcBr}, {"AcBr", 1}}},
            {"B", {{"AcBr", BcCc}, {"BcCc", 1}}},
            {"bias", {{"BcCc", 1}}}};

        // C += A * B (with zero-init)
        // followed by relu(C + bias) before storing
        auto mm = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A", "B"}, "C", strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, std::nullopt, nullptr, {},
            compose(elementwise_bias<DABUN_ISA>, elementwise_relu<DABUN_ISA>),
            {"bias"});

        // add for-loops from the order
        auto curr = mm;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            curr = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                it->first, it->second, {curr});
        }

        auto nodes = {curr};
        auto tree  = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            nodes, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
        ----------> Execution
        */
        auto A = get_random_vector<float_t>(AcBr * ArCr);
        auto B = get_random_vector<float_t>(AcBr * BcCc);

        auto CN = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ = CN;

        auto bias = get_random_vector<float_t>(1 * BcCc * 1);

        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 0);
        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        apply_relu(CN.data(), CN.data() + ArCr * BcCc);

        fn({{"C", CJ.data()},
            {"A", A.data()},
            {"B", B.data()},
            {"bias", bias.data()}});

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";
    }

    {
        // Matrix multiplication without zero init
        // bias and relu as pre-op

        /*
        ----------> Definition
        */
        int ArCr = 100;
        int AcBr = 100;
        int BcCc = 100;

        // schedule
        std::vector<std::pair<std::string, int>> order = {
            {"AcBr", 256}, {"ArCr", 3}, {"BcCc", 16}, {"AcBr", 1},
            {"AcBr", 1},   {"ArCr", 1}, {"BcCc", 1}};

        // sizes of dimensions
        std::map<std::string, int> sizes = {
            {"AcBr", AcBr}, {"ArCr", ArCr}, {"BcCc", BcCc}};

        // dimensions in each tensor
        std::map<std::string, std::set<std::string>> formulas = {
            {"C", {"ArCr", "BcCc"}},
            {"A", {"ArCr", "AcBr"}},
            {"B", {"AcBr", "BcCc"}}};

        // strides defining innermost computation
        std::map<std::string, std::map<std::string, int>> strides = {
            {"C", {{"ArCr", BcCc}, {"BcCc", 1}}},
            {"A", {{"ArCr", AcBr}, {"AcBr", 1}}},
            {"B", {{"AcBr", BcCc}, {"BcCc", 1}}},
            {"bias", {{"BcCc", 1}}}};

        // C += A * B (without zero-init)
        // followed by relu(C + bias) before storing
        auto mm = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A", "B"}, "C", strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 1, std::nullopt,
            compose(elementwise_bias<DABUN_ISA>, elementwise_relu<DABUN_ISA>),
            {"bias"}, nullptr, {});

        // add for-loops from the order
        auto curr = mm;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            curr = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                it->first, it->second, {curr});
        }

        auto nodes = {curr};
        auto tree  = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            nodes, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
        ----------> Execution
        */
        auto A    = get_random_vector<float_t>(AcBr * ArCr);
        auto B    = get_random_vector<float_t>(AcBr * BcCc);
        auto CN   = get_random_vector<float_t>(ArCr * BcCc);
        auto CJ   = CN;
        auto bias = get_random_vector<float_t>(1 * BcCc * 1);

        baseline_matrix_bias(ArCr, BcCc, BcCc, 1, 0, 1, CN.data(), bias.data());
        apply_relu(CN.data(), CN.data() + ArCr * BcCc);
        baseline_MM(ArCr, AcBr, BcCc, AcBr, 1, BcCc, 1, BcCc, 1, A.data(),
                    B.data(), CN.data(), 1);

        fn({{"C", CJ.data()},
            {"A", A.data()},
            {"B", B.data()},
            {"bias", bias.data()}});

        std::cout << "MAXABSDIFF: "
                  << max_abs_difference(CJ.data(), CJ.data() + ArCr * BcCc,
                                        CN.data())
                  << "\n";
    }

    {
        // another convolution example with relu as post op

        /*
        ----------> Definition
        */
        int OX = 101;
        int OY = 101;
        int OZ = 16 * 12 + 3;
        int KX = 3;
        int KY = 3;
        int KZ = 3;
        int IX = OX + KX - 1;
        int IY = OY + KY - 1;
        int IZ = OZ + KZ - 1;

        // schedule
        std::vector<std::pair<std::string, int>> order = {
            {"OX", 1}, {"OY", 10}, {"OY", 1}, {"OZ", 16},
            {"KX", 1}, {"KY", 1},  {"KZ", 1}, {"OZ", 1}};

        // sizes of dimensions
        std::map<std::string, int> sizes = {{"OX", OX}, {"OY", OY}, {"OZ", OZ},
                                            {"KX", KX}, {"KY", KY}, {"KZ", KZ}};

        // dimensions in each tensor
        std::map<std::string, std::set<std::string>> formulas = {
            {"C", {"OX", "OY", "OZ"}},
            {"A", {"IX", "IY", "IZ"}},
            {"B", {"KX", "KY", "KZ"}}};

        // strides defining innermost computation
        std::map<std::string, std::map<std::string, int>> strides = {
            {"C", {{"OX", OY * OZ}, {"OY", OZ}, {"OZ", 1}}},
            {"A",
             {{"OX", IY * IZ},
              {"OY", IZ},
              {"OZ", 1},
              {"KX", IY * IZ},
              {"KY", IZ},
              {"KZ", 1}}},
            {"B", {{"KX", KY * KZ}, {"KY", KZ}, {"KZ", 1}}}};

        // innermost op with zero init, and relu postop
        auto mm = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A", "B"}, "C", strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0, std::nullopt, nullptr, {},
            elementwise_relu<DABUN_ISA>);

        // add for-loops from the order
        auto curr = mm;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            curr = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                it->first, it->second, {curr});
        }

        auto nodes = {curr};
        auto tree  = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            nodes, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
        ----------> Execution
        */
        auto A  = get_random_vector<float_t>(IX * IY * IZ);
        auto B  = get_random_vector<float_t>(KX * KY * KZ);
        auto CN = std::vector<float_t>(OX * OY * OZ);
        auto CJ = std::vector<float_t>(OX * OY * OZ);

        if (!SKIP_EXPENSIVE)
        {
            baseline_3DConv(OX, OY, OZ, KX, KY, KZ, A.data(), B.data(),
                            CN.data());
            apply_relu(CN.data(), CN.data() + CN.size());

            fn({{"C", CJ.data()}, {"A", A.data()}, {"B", B.data()}});

            std::cout << "MAXABSDIFF: "
                      << max_abs_difference(CJ.data(), CJ.data() + OX * OY * OZ,
                                            CN.data())
                      << "\n";
        }
    }
#endif

    {
        // another convolution example

        /*
        ----------> Definition
        */
        int CIN  = 128;
        int COUT = 128 + 3;
        int OS   = 56 + 4;
        int KS   = 3;
        int IS   = OS + KS - 1;

        // schedule
        std::vector<std::pair<std::string, int>> order = {
            {"c_out", 16}, {"o_h", 1}, {"o_w", 28}, {"c_in", 16}, {"c_in", 1},
            {"o_w", 1},    {"k_h", 1}, {"k_w", 1},  {"c_out", 1}};

        // sizes of dimensions
        std::map<std::string, int> sizes = {{"c_out", COUT}, {"o_w", OS},
                                            {"k_w", KS},     {"c_in", CIN},
                                            {"o_h", OS},     {"k_h", KS}};

        // dimensions in each tensor
        std::map<std::string, std::set<std::string>> formulas = {
            {"C", {"c_out", "o_w", "o_h"}},
            {"A", {"c_in", "i_w", "i_h"}},
            {"B", {"c_in", "c_out", "k_w", "k_h"}}};

        // strides defining innermost computation
        std::map<std::string, std::map<std::string, int>> strides = {
            {"C", {{"o_w", COUT}, {"c_out", 1}, {"o_h", COUT * OS}}},
            {"A",
             {{"o_w", CIN},
              {"k_w", CIN},
              {"c_in", 1},
              {"o_h", IS * CIN},
              {"k_h", IS * CIN}}},
            {"B",
             {{"c_out", 1},
              {"c_in", COUT},
              {"k_w", COUT * CIN},
              {"k_h", COUT * CIN * KS}}}};

        // innermost op with zero init
        auto mm = make_compute_node<DABUN_VEX, DABUN_ARITHMETIC>(
            {"A", "B"}, "C", strides, arithmetic_op_kind::plus,
            arithmetic_op_kind::multiplies, 0);

        // add for-loops from the order
        auto curr = mm;
        for (auto it = order.rbegin(); it != order.rend(); it++)
        {
            curr = make_for_loop_node<DABUN_VEX, DABUN_ARITHMETIC>(
                it->first, it->second, {curr});
        }

        auto nodes = {curr};
        auto tree  = make_loop_tree_program<DABUN_VEX, DABUN_ARITHMETIC>(
            nodes, sizes, formulas, MAX_INTERPRETED_DEPTH);

        auto fn = tree->get_fn();

        /*
        ----------> Execution
        */
        auto A  = get_random_vector<float_t>(CIN * IS * IS);
        auto B  = get_random_vector<float_t>(COUT * CIN * KS * KS);
        auto CN = std::vector<float_t>(COUT * OS * OS);
        auto CJ = std::vector<float_t>(COUT * OS * OS);

        if (!SKIP_EXPENSIVE)
        {
            baseline_Conv(COUT, CIN, OS, OS, KS, KS, A.data(), B.data(),
                          CN.data());

            fn({{"C", CJ.data()}, {"A", A.data()}, {"B", B.data()}});

            std::cout << "MAXABSDIFF: "
                      << max_abs_difference(
                             CJ.data(), CJ.data() + COUT * OS * OS, CN.data())
                      << "\n";
        }
    }
}
