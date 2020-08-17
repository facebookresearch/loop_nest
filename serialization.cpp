#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "serialization.h"

int main()
{
    int  ArCr = 256;
    int  AcBr = 256;
    int  BcCc = 256;
    auto s    = facebook::sysml::aot::serialized_loop_nest_inputs(
        // The first argument is the loop order in the form of
        // {dimension, stride}.  For now the outer dimension
        // has to divide the stride.  This is effectively the
        // same as Halide's split into outer and inner
        // variable, but can have arbitray number of splits.
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
        {{"AcBr", BcCc}, {"BcCc", 1}}, 1024);
    auto str_rep = s.str();
    std::cout << str_rep << std::endl;

    auto s2       = facebook::sysml::aot::serialized_loop_nest_inputs::from_str(str_rep);
    auto str_rep2 = s2.str();
    std::cout << str_rep2 << std::endl;

    std::ofstream out("jose_test.txt");
    out << str_rep2;
    out.close();

    auto s3 = facebook::sysml::aot::serialized_loop_nest_inputs::from_file(
        "jose_test.txt");
    auto str_rep3 = s3.str();
    std::cout << str_rep3 << std::endl;
}