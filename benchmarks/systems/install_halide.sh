#!/usr/bin/env bash

# Installing Halide

HOME=/mnt/ssd1/josepablocam/
popd ${HOME}

# Install LLVM dependency (from source so we can turn off assertions)
git clone https://github.com/llvm/llvm-project.git --depth 1 -b release/10.x llvm-project-halide
mkdir llvm-build-halide
pushd llvm-build-halide

# build LLVM with assertions off (important for compile time)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../llvm-install-halide \
        -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" \
        -DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Mips;Hexagon" \
        -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF \
        ../llvm-project-halide/llvm

cmake --build . --target install
popd

export LLVM_CONFIG=${HOME}/llvm-install-halide/bin/llvm-config

git clone https://github.com/halide/Halide.git
pushd Halide
make
# run jit tests
make run_tests
popd

popd
