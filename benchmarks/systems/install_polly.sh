#!/usr/bin/env bash

# Installing Polly fork used in 
# *High-Performance Generalized Tensor Operations: A Compiler-Oriented Approach* (Gareev et al) (TACO 2018)
# https://bitbucket.org/gareevroman/polly-groman-fork/src/groman-fork/

HOME=/mnt/ssd1/josepablocam/

pushd ${HOME}

# Copy this particular fork of polly
git clone https://bitbucket.org/gareevroman/polly-groman-fork.git polly

# Install LLVM dependency
git clone https://github.com/llvm/llvm-project.git llvm-project-polly
pushd llvm-project-polly

# specific version required by this fork of polly
git checkout 217704f7a88244b6fc63008dc4518bf2cf2b3301
git checkout -b building-polly

# Replace LLVM's polly directory with the fork of polly
# TODO: for some reason replacing polly is actually breaking things
# despite the instructions from the paper's repo
# so for now we're technically using standard Polly...
# and using it from the main polly repo (not this checkedout version)
# rm -rf polly/
# cp -r ../polly/ polly/
popd

mkdir llvm-build-polly
pushd llvm-build-polly
cmake -DLLVM_ENABLE_PROJECTS='polly;clang' ../llvm-project-polly/llvm && make
# check if install worked
make check-polly

popd






