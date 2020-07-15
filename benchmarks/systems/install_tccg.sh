#!/usr/bin/env bash

# Installing TCCG (https://dl.acm.org/doi/pdf/10.1145/3157733)
# https://github.com/HPAC/tccg

HOME=/mnt/ssd1/josepablocam/
cd ${HOME}

# Requires TCL
wget https://prdownloads.sourceforge.net/tcl/tcl8.6.10-src.tar.gz
gunzip tcl8.6.10-src.tar.gz
tar xvf tcl8.6.10-src.tar
pushd tcl8.6.10/unix
./configure --prefix=${HOME}/tcl8
make
make install
popd
export TCL_ROOT=${HOME}/tcl8/
ln -s ${TCL_ROOT}/lib/libtcl8.6.so ${TCL_ROOT}/lib/libtcl.so


# Requires HPTT (transposer library)
git clone https://github.com/springer13/hptt.git
pushd hptt 
export CXX=g++
make avx
popd

export HPTT_ROOT=${HOME}/hptt
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/hptt/lib
export LD_LIBRARY_PATH


# Requires MKL 
# Using version already set up on AWS machine
mkdir -p ${HOME}/mkl/
ln -s /opt/intel/mkl/lib/intel64 ${HOME}/mkl/lib
ln -s /opt/intel/mkl/include/ ${HOME}/mkl/include
export MKLROOT=${HOME}/mkl/

# Install TCCG
git clone https://github.com/HPAC/tccg.git
pushd tccg
# Copy over our patched Makefile
# includes missing libraries and includes
cp -y ../tccg_Makefile tccg/Makefile
python setup.py install --user
export TCCG_ROOT=`pwd`
popd

# # Example usage
# scripts/tccg \
#   --arch=avx2 --numThreads=1 \
#   --floatType=s --compiler=g++ --verbose \
#   --ignoreDatabase \
#   --noGEMM \
#   example.tccg


