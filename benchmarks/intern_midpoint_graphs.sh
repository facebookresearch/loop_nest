#!/usr/bin/env bash

OUTDIR="intern-slides/"
mkdir  ${OUTDIR}

# Halide/loop_nest compile times
python3 utils/barplot_benchmarks.py \
	--input edge-benchmarks/*results/summary.csv \
	--field compile \
	--denominator "loop_nest" \
	--ylabel "Compile Time Ratio" \
	--title "Relative Compile Time" \
	--output ${OUTDIR}/halide_compile.pdf


# Halide/loop_nest running time
python3 utils/barplot_benchmarks.py \
	--input edge-benchmarks/*results/summary.csv \
	--field gflops \
	--denominator "halide" \
	--ylabel "GFLOPS Ratio" \
	--title "Relative GFLOPS" \
	--add_unit_line \
	--output ${OUTDIR}/halide_runtime.pdf

# GEMM
python3 utils/barplot_benchmarks.py \
	--input mm-benchmarks/*results/summary.csv \
	--field gflops \
	--ylabel GFLOPS \
	--title "Comparison MM" \
	--filter "df[df['plus_op'] == '+']" \
	--output ${OUTDIR}/gemm_runtime.pdf



# DNN ops
python3 utils/barplot_benchmarks.py \
	--input operator-benchmarks/*results/summary.csv \
	--field gflops \
	--output ${OUTDIR}/operator_runtime.pdf


tar cvf intern-slides.tar intern-slides/
