#!/usr/bin/env bash

for f in *-benchmarks
do
	echo "Removing results in $f"
	pushd $f
	rm -rf *results
	popd
done
