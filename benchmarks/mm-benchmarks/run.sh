#!/usr/bin/env bash
HOME=/mnt/ssd1/josepablocam

SIZES="32 64 128 256 512"
ARCHES="avx2 avx512"

mkdir tccg-results/
source ../systems/tccg_vars.sh
python3 run_tccg.py \
	--tccg ${HOME}/tccg/ \
	--size ${SIZES} \
	--arch ${ARCHES} \
	--output_dir tccg-results/
# intermediate files produced by tccg
rm -rf tccg_implementations/

mkdir polly-results/
source ../systems/polly_vars.sh
python3 run_polly.py \
	--llvm ${HOME}/llvm-build-polly/ \
	--size ${SIZES} \
	--arch ${ARCHES} \
	--plus_op + max max max min min\
	--multiplies_op "*" + min "*" "*" + \
	--output_dir polly-results/


mkdir loop-nest-results/
python3 run_loop_nest.py \
        --loop_nest ${HOME}/loop_nest/ \
	--cpp loop_nest_matmul.cpp \
        --size ${SIZES} \
	--arch ${ARCHES} \
        --plus_op + max max max min min\
        --multiplies_op "*" + min "*" "*" + \
        --output_dir loop-nest-results/









