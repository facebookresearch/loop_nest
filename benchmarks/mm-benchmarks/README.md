# Matrix-Matrix Operations

Based on experiments in "High-Performance Generalized Tensor Operations: A Compiler-Oriented Approach" (Gareev et al, TACO 2018)
https://dl.acm.org/doi/pdf/10.1145/3235029.

They perform matrix-matrix operations using:

* +, max
* /, max
* min, max
* x, max
* x, min
* x, -
* x, min

All dimensions in are equal (i.e. m = n = k)
and they vary the sizes from 32 to 4000).

For our experiments, we cover:
* +, max
* min, max
* x, max
* x, min
* x, min

Sizes: 32, 64, 128, 256, 512

We compare:
* loop_nest
* Polly-based compiler (https://bitbucket.org/gareevroman/polly-groman-fork/src/groman-fork/docs/UsingPollyWithClang.rst)
* TCCG (https://github.com/HPAC/tccg)

TODO:
* Using the latest polly, rather than the one from the paper.
It looks like the latest polly has these changes merged (based on their performance graph),
but currently performance is bad for everything except standard matrix multiplication
(despite the paper showing different operators)
* loop_nest should use our own search for schedules rather than hard-coded schedule
