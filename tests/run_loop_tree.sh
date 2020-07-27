#!/usr/bin/env bash
# Run from root of loop nest folder

# with jitting etc
g++ -g -Wall -Wpedantic -std=c++17 loop_tree.cpp \
-I ./xbyak -I ./ \
-Wno-sign-compare \
-DCT_ISA=avx2  \
-DNDEBUG=1 -o test.out && ./test.out

# without jitting
g++ -g -Wall -Wpedantic -std=c++17 loop_tree.cpp \
-I ./xbyak -I ./ \
-Wno-sign-compare \
-DCT_ISA=avx2  \
-DNDEBUG=1 \
-DNOPTIM \
-o test.out && ./test.out

# only jitting part of the loop nest (top levels interpreted)
g++ -g -Wall -Wpedantic -std=c++17 loop_tree.cpp \
-I ./xbyak -I ./ \
-Wno-sign-compare \
-DCT_ISA=avx2  \
-DNDEBUG=1 \
-DNOPTIM \
-DTEST_STOP_SIMPLIFICATION \
-o test.out && ./test.out
