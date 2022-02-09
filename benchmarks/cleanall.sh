#!/usr/bin/env bash

for f in ./*-benchmarks
do
	echo "Removing results in ${f}"
	pushd "${f}" || exit 1
	rm -rf ./*results
	popd || exit 1
done
