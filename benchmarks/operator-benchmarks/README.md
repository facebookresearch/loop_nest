# NN Operations

Based on experiments in 
"FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System"
(Zheng et al, ASPLOS 2020)
https://dl.acm.org/doi/abs/10.1145/3373376.3378508


They perform the following operations (with varying sizes):
* GEMV
* GEMM
* Bilinear
* 1D convolution
* Transposed 1D convolution (i.e. "deconvolution")
* 2D convolution
* Transposed 2D convolution 
* 3D convolution
* Transposed 3D convolution
* Group convolution
* Depthwise convolution
* Dilated convolution


For our experiments, we cover:
* GEMV
* GEMM
* 1D Convolution (TODO)
* 2D Convolution
* 3D Convolution
* Dilated Convolution (TODO)

We compare:

* loop_nest
* Halide
* oneDNN (formerly MKL-DNN) (https://github.com/oneapi-src/oneDNN)

TODO:
* Halide needs to use Buffer not ImageParam (which is slower)
* Halide should be built with LLVM assertions turned off
* remaining operators
* oneDNN
* Halide should use autotuning
* loop_nest should use our own search (rather than hardcoded schedules)
* consider adding pre/post operations to these (currently none used)
