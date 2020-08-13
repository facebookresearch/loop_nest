# Edge Benchmarks

Collection of edge-case benchmarks (e.g. weird schedules, shapes)
originally written as part of test examples for loop_nest.


Systems:
* LLVM (w/ Halide)
* loop_nest

TODOs:
* If we want to have post-ops in these benchmarks, we need
to fix the way the translate_to_halide tool handles post-ops,
so that they are not generated as a whole additional pass
over the output tensor
