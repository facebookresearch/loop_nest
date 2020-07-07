#!/usr/bin/env bash

export HALIDE_PATH=~/Halide
export LD_LIBRARY_PATH=${HALIDE_PATH}/bin

SRC_DIR="../"

# make sure halide flags are set off
unset HL_DEBUG_CODEGEN

run_halide() {
    avx=$1
    echo "Running Halide:${avx}"

    g++ ${SRC_DIR}/translate_to_halide.cpp -g  \
    -I ${HALIDE_PATH}/include -I${SRC_DIR}/xbyak  \
    -L ${HALIDE_PATH}/bin -lHalide -lpthread -ldl  -std=c++17   \
    -DCT_ISA="${avx}" \
    -O3 \
    -o "translate_to_halide_${avx}.out" \
    && "./translate_to_halide_${avx}.out" > "halide_${avx}_results.txt"
}

run_loop_nest() {
    avx=$1
    echo "Running loop_nest:${avx}"

    # make sure to run loop_nest without logging messages
    # since we time code generation as well...
    g++ -Wall -Wpedantic  \
    -std=c++17 ${SRC_DIR}/loop_nest.cpp \
    -I${SRC_DIR}/xbyak \
    -Wno-sign-compare \
    -DCT_ISA="${avx}" \
    -DNDEBUG=1 \
    -O3 \
    -o "loop_nest_${avx}.out" \
    && "./loop_nest_${avx}.out" > "loop_nest_${avx}_results.txt"
}


run_halide avx2
run_loop_nest avx2

python3 barplot_benchmarks.py \
    --input halide_avx2_results.txt loop_nest_avx2_results.txt \
    --label Halide loop_nest \
    --output avx2_results.pdf



run_halide avx512
run_loop_nest avx512

python3 barplot_benchmarks.py \
    --input halide_avx512_results.txt loop_nest_avx512_results.txt \
    --label Halide loop_nest \
    --output avx512_results.pdf
