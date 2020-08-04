#!/usr/bin/env bash

export HALIDE_PATH=~/halide_build
export LD_LIBRARY_PATH=${HALIDE_PATH}/bin

SRC_DIR="../"

# make sure halide flags are set off
unset HL_DEBUG_CODEGEN

run_halide() {
    avx=$1
    echo "Running Halide:${avx}"

    g++ ${SRC_DIR}/translate_to_halide_arm.cpp -g  \
    -I ${HALIDE_PATH}/include -I${SRC_DIR}/xbyak_aarch64  \
    -L ${HALIDE_PATH}/bin -lHalide -lpthread -ldl -DLOOP_NEST_ARM -std=c++17   \
    -DCT_ISA="${avx}" \
    -O3 \
    -o "translate_to_halide_${avx}.out" \
    && numactl -C 5 "./translate_to_halide_${avx}.out" > "halide_${avx}_results.txt"
}

run_loop_nest() {
    avx=$1
    echo "Running loop_nest:${avx}"

    # make sure to run loop_nest without logging messages
    # since we time code generation as well...
    g++ -Wall -Wpedantic  \
    -std=c++17 ${SRC_DIR}/loop_nest_arm.cpp \
    -I${SRC_DIR}/xbyak_aarch64 \
    -Wno-sign-compare \
    -DCT_ISA="${avx}" \
    -DNDEBUG=1 \
    -DLOOP_NEST_ARM \
    -O3 \
    -o "loop_nest_${avx}.out" \
    && numactl -C 5 "./loop_nest_${avx}.out" > "loop_nest_${avx}_results.txt"
}


run_halide aarch64
run_loop_nest aarch64

python3 barplot_benchmarks.py \
    --input halide_aarch64_results.txt loop_nest_aarch64_results.txt \
    --label Halide loop_nest \
    --output aarch64_results.pdf
