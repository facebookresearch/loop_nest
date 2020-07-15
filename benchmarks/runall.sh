#!/usr/bin/env bash

echo "Running *all* benchmarks"
for bf in *-benchmarks
do
	echo "Running ${bf}"
	pushd ${bf}
	bash run.sh
	popd 
done
