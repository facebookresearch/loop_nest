#!/usr/bin/env bash

# Installing TCCG (https://dl.acm.org/doi/pdf/10.1145/3157733)
# https://github.com/HPAC/tccg

HOME=/mnt/ssd1/josepablocam/

SYSTEMS_DIR=$(pwd)

pushd ${HOME} || exit 1

# Requires TCL
wget https://prdownloads.sourceforge.net/tcl/tcl8.6.10-src.tar.gz
gunzip tcl8.6.10-src.tar.gz
tar xvf tcl8.6.10-src.tar
pushd tcl8.6.10/unix || exit 1
./configure --prefix=${HOME}/tcl8
make
make install
popd || exit 1
export TCL_ROOT="${HOME}/tcl8/"
# to match tccg makefile naming
ln -s "${TCL_ROOT}/lib/libtcl8.6.so" "${TCL_ROOT}/lib/libtcl.so"


# Requires HPTT (transposer library)
git clone https://github.com/springer13/hptt.git
pushd hptt || exit 1
export CXX=g++
make avx
popd || exit 1

export HPTT_ROOT="${HOME}/hptt"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"${HOME}/hptt/lib"
export LD_LIBRARY_PATH


# Requires MKL
# Using version already set up on AWS machine
# Just matching tccg makefile assumed paths
mkdir -p "${HOME}/mkl/"
ln -s /opt/intel/mkl/lib/intel64 "${HOME}/mkl/lib"
ln -s /opt/intel/mkl/include/ "${HOME}/mkl/include"
export MKLROOT="${HOME}/mkl/"

# Install TCCG
git clone https://github.com/HPAC/tccg.git
pushd tccg || exit 1
# Copy over our patched Makefile
# includes missing libraries and includes
cp -y "${SYSTEMS_DIR}/tccg_Makefile" tccg/Makefile
python setup.py install --user
TCCG_ROOT=$(pwd)
export TCCG_ROOT
popd || exit 1


popd || exit 1

# # Example usage
# pushd ${HOME}/tccg
# scripts/tccg \
#   --arch=avx2 --numThreads=1 \
#   --floatType=s --compiler=g++ --verbose \
#   --ignoreDatabase \
#   --noGEMM \
#   example.tccg
# popd
