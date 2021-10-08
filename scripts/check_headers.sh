#!/bin/bash

for value in $(find ./include | grep "\.hpp")
do
    echo -n "#include \"" > tmp_to_compile.cpp
    echo -n $value >> tmp_to_compile.cpp
    echo "\"" >> tmp_to_compile.cpp
    echo "int main() {}" >> tmp_to_compile.cpp
    g++ -g -std=c++2a tmp_to_compile.cpp -I./third-party -I./include -I./extern/xbyak_aarch64 -I./extern/fmt/include  -DDABUN_ISA=aarch64 -DDABUN_ARITHMETIC=dabun::fp16  -DDABUN_ARM -I../../boost_1_75_0 -o tmp_to_compile
done

rm -f tmp_to_compile*
