#!/usr/bin/env bash
HOME=/mnt/ssd1/josepablocam

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]
then
    echo "Usage: <folder-with-serialized-inputs> [limit]"
    exit 1
fi

SERIALIZED_INPUTS=$(ls $1/*.txt)
LIMIT=-1

if [ "$#" -eq 2 ]
then
    LIMIT=$2
fi

SEED=42
ARCHES[0]="avx2"
ARCHES[1]="avx512"


# shellcheck disable=SC1091
source ../systems/halide_vars.sh
mkdir halide-from-file-results
python3 run_halide.py \
    --loop_nest ../../ \
	--halide "${HALIDE_PATH}" \
	--cpp halide_from_file.cpp \
    --serialized_inputs ${SERIALIZED_INPUTS} \
	--arch "${ARCHES[@]}" \
    --output_dir halide-from-file-results \
    --limit ${LIMIT} \
    --seed ${SEED}


mkdir loop-nest-from-file-results/
python3 run_loop_nest.py \
    --loop_nest ../../ \
	--cpp loop_nest_from_file.cpp \
    --serialized_inputs ${SERIALIZED_INPUTS} \
	--arch "${ARCHES[@]}" \
    --output_dir loop-nest-from-file-results/ \
    --limit ${LIMIT} \
    --seed ${SEED}
