#!/usr/bin/env python3
from argparse import ArgumentParser
import csv
import os
import random
import subprocess

DEFAULT_EXECUTABLE_NAME = "test.out"

def compile_halide(loop_nest_root, halide_root, arch, input_path, output_path, exec_name=None):
    if exec_name is None:
        exec_name = DEFAULT_EXECUTABLE_NAME

    print("Compiling", input_path)
    cmd = [
        "g++",
        input_path,
        "-std=c++17",
        "-I {}/xbyak".format(loop_nest_root),
        "-I {}".format(loop_nest_root),
        "-I {}/include".format(halide_root),
        "-L {}/bin".format(halide_root),
        "-lHalide",
        "-lpthread",
        "-ldl",
        "-O3",
        "-DCT_ISA={}".format(arch),
        "-o",
        exec_name
    ]
    with open(output_path, "w") as fout:
        proc = subprocess.run(" ".join(cmd), shell=True, stdout=fout)
    return proc.returncode

def run_halide(output_path, input_path=None, exec_name=None):
    if exec_name is None:
        exec_name = DEFAULT_EXECUTABLE_NAME

    cmd = ["./" + exec_name, ]
    if input_path is not None:
        print(input_path)
        cmd.append(input_path)

    with open(output_path, "a") as fout:
        proc = subprocess.run(" ".join(cmd), shell=True, stdout=fout)

    return proc.returncode

def cleanup(exec_name=None):
    if exec_name is None:
        exec_name = DEFAULT_EXECUTABLE_NAME
    os.remove(exec_name)


def extract_results(output_path):
    results = []
    with open(output_path, "r") as fin:
        entry = []
        benchmark_name = None
        for line in fin:
            if line.startswith("Benchmark:"):
                benchmark = line.split(":")[-1].strip()
                if benchmark_name is not None:
                    # a benchmark failed (so don't keep accumulating)
                    entry = []
                benchmark_name = benchmark
                entry.append(benchmark)
            if line.startswith("Compile"):
                compile_time = float(line.split(":")[-1].strip())
                assert compile_time > 0
                entry.append(compile_time)
            if line.startswith("GFLOPS"):
                gflops = float(line.split(":")[-1].strip())
                assert gflops > 0
                entry.append(gflops)
                assert len(entry) == 3
                results.append(entry)
                entry = []
                benchmark_name = None

    return results


def dump_summary(summary_log, output_path):
    defaults = ("llvm", )

    with open(output_path, "w") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow(["system", "isa", "benchmark", "compile", "gflops"])
        for entry in summary_log:
            writer.writerow(defaults + entry)


def run_experiments(loop_nest_root, halide_root, cpp_file, archs, output_dir, serialized_inputs=None):
    summary_log = []
    for arch in archs:
        output_path = os.path.join(output_dir, "dnn-ops-{}.txt".format(arch))
        print("Running Halide on", output_path)
        ret = compile_halide(loop_nest_root, halide_root, arch, cpp_file, output_path)
        if ret != 0:
            print("Halide failed compiling on", output_path)
        if serialized_inputs is None:
            ret = run_halide(output_path)
            if ret != 0:
                print("Halide failed running on", output_path)
        else:
            for input_path in serialized_inputs:
                ret = run_halide(output_path, input_path)
                if ret != 0:
                    print("Halide failed running on", input_path)
        results = extract_results(output_path)
        results = [tuple([arch] + entry) for entry in results]
        summary_log.extend(results)
    
    cleanup()
    dump_summary(summary_log, os.path.join(output_dir, "summary.csv"))


def get_args():
    parser = ArgumentParser(
        description="Run NN operator benchmarks for LLVM w/Halide")
    parser.add_argument("--loop_nest",
                        type=str,
                        help="Path to loop_nest root directory")
    parser.add_argument("--halide",
                        type=str,
                        help="Path to Halide root directory")
    parser.add_argument("--cpp",
                        type=str,
                        help="Path to cpp file with harness code")
    parser.add_argument("--serialized_inputs", type=str, nargs="+", help="Benchmarks as serialized loop_nest inputs")
    parser.add_argument("--limit", type=int, help="Limit on number of serialized inputs to run (randomly sampled)")
    parser.add_argument("--seed", type=int, help="Random number generator", default=42)
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

    serialized_inputs = args.serialized_inputs
    if args.limit is not None and len(serialized_inputs) > args.limit:
        print("Downsampling serialized inputs to", args.limit)
        random.seed(args.seed)
        random.shuffle(serialized_inputs)
        serialized_inputs = serialized_inputs[:args.limit]
    
    run_experiments(args.loop_nest, args.halide, args.cpp, args.arch,
                    args.output_dir, serialized_inputs)


if __name__ == "__main__":
    try:
        main()
    except Exception as err: # noqa F841
        import pdb
        pdb.post_mortem()
