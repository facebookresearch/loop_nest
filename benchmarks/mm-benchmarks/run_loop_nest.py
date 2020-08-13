#!/usr/bin/env python3
from argparse import ArgumentParser
import csv
import os
import subprocess


def get_op(op):
    if op == "+":
        return "basic_plus"
    elif op == "*":
        return "basic_multiplies"
    elif op == "max" or op == "min":
        return op
    else:
        raise Exception("Unhandled op={}".format(op))


def run_loop_nest(loop_nest_root, size, arch, op_pair, input_path,
                  output_path,):
    cmd = [
        "g++", "-Wall", "-Wpedantic", "-Wno-sign-compare", "-std=c++17",
        "-I {}/xbyak".format(loop_nest_root), "-I {}".format(loop_nest_root),
        "-O3", "-DNDEBUG=1", "-DCT_ISA={}".format(arch),
        "-DPLUS_OP={}".format(get_op(op_pair[0])),
        "-DMULTIPLIES_OP={}".format(get_op(op_pair[1])), input_path, "-o",
        "test.out", "&&", "./test.out",
        str(size),
    ]

    with open(output_path, "w") as fout:
        proc = subprocess.run(" ".join(cmd), shell=True, stdout=fout)

    os.remove("test.out")
    return proc.returncode


def extract_results(output_path):
    compile_time = -1
    gflops = -1

    with open(output_path, "r") as fin:
        for line in fin:
            if line.startswith("Compile"):
                compile_time = float(line.split(":")[-1].strip())
            if line.startswith("GFLOPS"):
                gflops = float(line.split(":")[-1].strip())
    assert compile_time > 0
    assert gflops > 0
    return compile_time, gflops


def dump_summary(summary_log, output_path):
    defaults = ("loop_nest", )

    with open(output_path, "w") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow([
            "system", "plus_op", "multiplies_op", "size", "isa", "benchmark",
            "compile", "gflops",
        ])
        for entry in summary_log:
            writer.writerow(defaults + entry)


def run_experiments(loop_nest_root, cpp_file, sizes, op_pairs, archs,
                    output_dir):
    summary_log = []
    for size in sizes:
        for op_pair in op_pairs:
            for arch in archs:
                output_path = os.path.join(
                    output_dir,
                    "mm-{}-{}-{}-{}.txt".format(size, op_pair[0], op_pair[1],
                                                arch))
                print("Running loop_nest on", output_path)
                ret = run_loop_nest(loop_nest_root, size, arch, op_pair,
                                    cpp_file, output_path,)
                if ret != 0:
                    print("loop_nest failed on", output_path)
                compile_time, gflops = extract_results(output_path)
                benchmark_name = "({})({}){}".format(op_pair[0], op_pair[1],
                                                     size,)
                summary_log.append((op_pair[0], op_pair[1], size, arch,
                                    benchmark_name, compile_time, gflops,))

    dump_summary(summary_log, os.path.join(output_dir, "summary.csv"))


def get_args():
    parser = ArgumentParser(description="Run MM benchmarks for loop_nest")
    parser.add_argument("--loop_nest",
                        type=str,
                        help="Path to loop_nest root directory")
    parser.add_argument("--cpp",
                        type=str,
                        help="Path to cpp file with benchmark code")
    parser.add_argument("--size", type=int, nargs="+", help="Sizes")
    parser.add_argument("--plus_op",
                        type=str,
                        nargs="+",
                        help="Plus operations")
    parser.add_argument("--multiplies_op",
                        type=str,
                        nargs="+",
                        help="Multiplies operations")
    parser.add_argument("--arch", type=str, nargs="+", help="ISAs")
    parser.add_argument("--output_dir",
                        type=str,
                        help="Output directory for results")
    return parser.parse_args()


def main():
    args = get_args()
    assert all(a in ["avx2", "avx512"]
               for a in args.arch), "Unsupported architecture"
    if not os.path.exists(args.output_dir):
        print("# mkdir -p {}".format(args.output_dir))
        os.makedirs(args.output_dir)

    assert len(args.plus_op) == len(args.multiplies_op)
    op_pairs = list(zip(args.plus_op, args.multiplies_op))

    run_experiments(args.loop_nest, args.cpp, args.size, op_pairs, args.arch,
                    args.output_dir,)


if __name__ == "__main__":
    try:
        main()
    except Exception as err: # noqa F841
        import pdb
        pdb.post_mortem()
