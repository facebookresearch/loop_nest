#!/usr/bin/env python3
from argparse import ArgumentParser
import csv
import os
import subprocess


def run_loop_nest(loop_nest_root, arch, input_path, output_path):
    cmd = [
        "g++",
        "-Wall",
        "-Wpedantic",
        "-Wno-sign-compare",
        "-std=c++17",
        "-I {}/xbyak".format(loop_nest_root),
        "-I {}".format(loop_nest_root),
        "-O3",
        "-DNDEBUG=1",
        "-DDABUN_ISA={}".format(arch),
        input_path,
        "-o",
        "test.out",
        "&&",
        "./test.out",
    ]

    with open(output_path, "w") as fout:
        proc = subprocess.run(" ".join(cmd), shell=True, stdout=fout)
    os.remove("test.out")
    return proc.returncode


def extract_results(output_path):
    results = []
    with open(output_path, "r") as fin:
        entry = []
        for line in fin:
            if line.startswith("Benchmark:"):
                benchmark = line.split(":")[-1].strip()
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

    return results


def dump_summary(summary_log, output_path):
    defaults = ("loop_nest", )

    with open(output_path, "w") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow(["system", "isa", "benchmark", "compile", "gflops"])
        for entry in summary_log:
            writer.writerow(defaults + entry)


def run_experiments(loop_nest_root, cpp_file, archs, output_dir):
    summary_log = []
    for arch in archs:
        output_path = os.path.join(output_dir, "edge-{}.txt".format(arch))
        print("Running loop_nest on", output_path)
        ret = run_loop_nest(loop_nest_root, arch, cpp_file, output_path)
        if ret != 0:
            print("loop_nest failed on", output_path)
        results = extract_results(output_path)
        results = [tuple([arch] + entry) for entry in results]
        summary_log.extend(results)

    dump_summary(summary_log, os.path.join(output_dir, "summary.csv"))


def get_args():
    parser = ArgumentParser(description="Run edge benchmarks for loop_nest")
    parser.add_argument("--loop_nest",
                        type=str,
                        help="Path to loop_nest root directory")
    parser.add_argument("--cpp",
                        type=str,
                        help="Path to cpp file with benchmark code")
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

    run_experiments(args.loop_nest, args.cpp, args.arch, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as err: # noqa F841
        import pdb
        pdb.post_mortem()
