#!/usr/bin/env bash
HOME=/mnt/ssd1/josepablocam

ARCHES="avx2 avx512"

mkdir loop-nest-results/
python3 run_loop_nest.py \
        --loop_nest ${HOME}/loop_nest/ \
	--cpp ../operator-benchmarks/loop_nest_nn_ops.cpp \
	--arch ${ARCHES} \
        --output_dir loop-nest-results/









