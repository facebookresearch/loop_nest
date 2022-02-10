#!/usr/bin/env bash

HOME=/mnt/ssd1/josepablocam
export HALIDE_PATH="${HOME}/Halide"
export LD_LIBRARY_PATH="${HALIDE_PATH}/bin"
# make sure halide runs single threaded (for fair comparison)
# with single-threaded loop_nest
export HL_NUM_THREADS=1
