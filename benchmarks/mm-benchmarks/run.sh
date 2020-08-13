#!/usr/bin/env bash
HOME=/mnt/ssd1/josepablocam

SIZES[0]=32
SIZES[1]=64
SIZES[2]=128
SIZES[3]=256
SIZES[4]=512

ARCHES[0]="avx2"
ARCHES[1]="avx512"

mkdir tccg-results/
# shellcheck disable=SC1091
source ../systems/tccg_vars.sh
python3 run_tccg.py \
	--tccg "${HOME}/tccg/" \
	--size "${SIZES[@]}" \
	--arch "${ARCHES[@]}" \
	--output_dir tccg-results/
# intermediate files produced by tccg
rm -rf tccg_implementations/

mkdir polly-results/
# shellcheck disable=SC1091
source ../systems/polly_vars.sh
python3 run_polly.py \
	--llvm "${HOME}/llvm-build-polly/" \
	--size "${SIZES[@]}" \
	--arch "${ARCHES[@]}" \
	--plus_op + max max max min min\
	--multiplies_op "*" + min "*" "*" + \
	--output_dir polly-results/


mkdir loop-nest-results/
python3 run_loop_nest.py \
    --loop_nest "${HOME}/loop_nest/" \
	--cpp loop_nest_matmul.cpp \
	--size "${SIZES[@]}" \
	--arch "${ARCHES[@]}" \
	--plus_op + max max max min min\
	--multiplies_op "*" + min "*" "*" + \
	--output_dir loop-nest-results/
