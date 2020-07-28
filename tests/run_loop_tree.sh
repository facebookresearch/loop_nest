#!/usr/bin/env bash
# Run from root of loop nest folder

# with jitting etc
echo "Jitting"
g++ -g -Wall -Wpedantic -std=c++17 loop_tree.cpp \
-I ./xbyak -I ./ \
-Wno-sign-compare \
-DCT_ISA=avx2  \
-o test.out 2>/dev/null && ./test.out | grep "MAXABSDIFF"

# without jitting
echo "No jitting"
g++ -g -Wall -Wpedantic -std=c++17 loop_tree.cpp \
-I ./xbyak -I ./ \
-Wno-sign-compare \
-DCT_ISA=avx2  \
-DNDEBUG=1 \
-DNOPTIM \
-DNELEMENTWISE \
-o test.out 2>/dev/null && ./test.out | grep "MAXABSDIFF"

# only jitting part of the loop nest (top levels interpreted)
echo "Partial jitting"
g++ -g -Wall \
-Wpedantic -std=c++17 \
loop_tree.cpp -I ./xbyak -I ./ \
-Wno-sign-compare -DCT_ISA=avx2  \
-DMAX_INTERPRETED_DEPTH=3 \
-o test.out 2>/dev/null && ./test.out | grep -e "loop_tree" -e "MAXABSDIFF"
