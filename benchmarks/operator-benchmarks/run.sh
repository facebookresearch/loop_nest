#!/usr/bin/env bash
HOME=/mnt/ssd1/josepablocam

ARCHES="avx2 avx512"

source ../systems/halide_vars.sh
mkdir halide-results
python3 run_halide.py \
        --loop_nest ${HOME}/loop_nest/ \
	--halide ${HALIDE_PATH} \
	--cpp halide_nn_ops.cpp \
	--arch ${ARCHES} \
        --output_dir halide-results/


mkdir loop-nest-results/
python3 run_loop_nest.py \
        --loop_nest ${HOME}/loop_nest/ \
	--cpp loop_nest_nn_ops.cpp \
	--arch ${ARCHES} \
        --output_dir loop-nest-results/









