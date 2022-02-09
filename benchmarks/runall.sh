#!/usr/bin/env bash

echo "Running *all* benchmarks"
for bf in ./*-benchmarks
do
	echo "Running ${bf}"
	pushd "${bf}" || exit 1
	bash run.sh
	popd || exit 1
done
