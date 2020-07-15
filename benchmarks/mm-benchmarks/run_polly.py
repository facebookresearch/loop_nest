#!/usr/bin/env python3
from argparse import ArgumentParser
import csv
import os
import subprocess
import tempfile

# based on
# https://bitbucket.org/gareevroman/polly-groman-fork/src/groman-fork/www/experiments/matmul/matmul.c
CPP_SHELL_CODE = """
#include <iostream>
#include <chrono>
#include <limits>


#define N {size}
float A[N][N];
float B[N][N];
float C[N][N];

void init_array()
{{
    int i, j;

    for (i=0; i<N; i++) {{
        for (j=0; j<N; j++) {{
            A[i][j] = (1+(i*j)%1024)/2.0;
            B[i][j] = (1+(i*j)%1024)/2.0;
        }}
    }}
}}


int main()
{{
    int i, j, k;

    init_array();

    auto start = std::chrono::high_resolution_clock::now();
        
    for(i=0; i<N; i++)  {{
        for(j=0; j<N; j++)  {{
            C[i][j] = {identity};
            for(k=0; k<N; k++) {{
		float temp = {multiplies_op};
		C[i][j] = {plus_op};
		}}
        }}
    }}
   
   auto end = std::chrono::high_resolution_clock::now();
   auto new_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
   std::cout << "Time: " << (static_cast<double>(new_time) / 1e9) << std::endl;
}}
"""


def get_identity_value(plus_op):
    if plus_op == "+":
        return "0"
    elif plus_op == "max":
        return "-std::numeric_limits<float>::infinity()"
    elif plus_op == "min":
        return "std::numeric_limits<float>::infinity()"
    else:
        raise Exception("Unhandled identity for plus_op={}".format(plus_op))


def get_op_str(op, left, right):
    if op == "*":
        return "{} * {}".format(left, right)
    elif op == "max":
        return "std::max({}, {})".format(left, right)
    elif op == "min":
        return "std::min({}, {})".format(left, right)
    elif op == "+":
        return "{} + {}".format(left, right)
    else:
        raise Exception("Unhandled op={}".format(multipies_op))


def get_multiplies_op_str(multiplies_op):
    left = "A[i][k]"
    right = "B[k][j]"
    return get_op_str(multiplies_op, left, right)


def get_plus_op_str(plus_op):
    left = "C[i][j]"
    right = "temp"
    return get_op_str(plus_op, left, right)


def generate_cpp_file(op_pair, size):
    plus_op, multiplies_op = op_pair
    code = CPP_SHELL_CODE.format(
        size=size,
        identity=get_identity_value(plus_op),
        multiplies_op=get_multiplies_op_str(multiplies_op),
        plus_op=get_plus_op_str(plus_op))
    input_file = tempfile.NamedTemporaryFile(mode="w",
                                             suffix=".cpp",
                                             delete=True)
    input_file.write(code)
    input_file.flush()
    return input_file


def run_polly(llvm_root, arch, input_path, output_path):
    arch = "avx512f" if arch == "avx512" else arch
    cmd = [
        os.path.join(llvm_root, "bin", "clang++"),
        "-O3",
        "-mllvm",
        "-polly",
        "-mllvm",
        "-polly-vectorizer=stripmine",
        "-m{}".format(arch),
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


def extract_seconds(output_path):
    with open(output_path, "r") as fin:
        log_line = fin.readlines()[-1]
        assert log_line.startswith("Time")
        log_line = log_line.replace("Time:", "")
        system_time = float(log_line.strip())
        assert system_time > 0
        return system_time


def extract_gflops(num_gflops, output_path):
    return num_gflops / extract_seconds(output_path)


def dump_summary(summary_log, output_path):
    defaults = ("polly", )

    with open(output_path, "w") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow([
            "system", "plus_op", "multiplies_op", "size", "isa", "benchmark",
            "gflops"
        ])
        for entry in summary_log:
            writer.writerow(defaults + entry)


def run_experiments(llvm_root_dir, sizes, op_pairs, archs, output_dir):
    summary_log = []
    for size in sizes:
        num_gflops = (size * size * size * 2.0) / 1e9
        for op_pair in op_pairs:
            input_file = generate_cpp_file(op_pair, size)
            for arch in archs:
                output_path = os.path.join(
                    output_dir,
                    "mm-{}-{}-{}-{}.txt".format(size, op_pair[0], op_pair[1],
                                                arch))
                print("Running", output_path)
                ret = run_polly(llvm_root_dir, arch, input_file.name,
                                output_path)
                if ret != 0:
                    print("Polly failed on", output_path)
                gflops = extract_gflops(num_gflops, output_path)
                benchmark_name = "({})({}){}".format(op_pair[0], op_pair[1],
                                                     size)
                summary_log.append((op_pair[0], op_pair[1], size, arch,
                                    benchmark_name, gflops))
        input_file.close()

    dump_summary(summary_log, os.path.join(output_dir, "summary.csv"))


def get_args():
    parser = ArgumentParser(description="Run MM benchmarks for Polly")
    parser.add_argument("--llvm", type=str, help="Path to llvm root directory")
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

    run_experiments(args.llvm, args.size, op_pairs, args.arch, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
