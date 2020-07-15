#!/usr/bin/env python3
from argparse import ArgumentParser
import csv
import os
import subprocess
import tempfile


def generate_cfg_file(size):
    code = """
    C[i,j] = A[i,k] * B[k,j]
    i = {}
    j = {}
    k = {}
    """.format(size, size, size)
    input_file = tempfile.NamedTemporaryFile(mode="w", delete=True)
    input_file.write(code)
    input_file.flush()
    return input_file


def run_tccg(tccg_root, arch, input_path, output_path):
    cmd = [
        os.path.join(tccg_root, "scripts", "tccg"),
        "--arch",
        arch,
        "--numThreads",
        "1",
        "--floatType",
        "s",
        "--compiler",
        "g++",
        "--verbose",
        "--ignoreDatabase",
        "--noGEMM",  # this is not part of tccg
        input_path,
    ]
    with open(output_path, "w") as fout:
        proc = subprocess.run(cmd, stdout=fout)

    return proc.returncode


def extract_gflops(output_path):
    with open(output_path, "r") as fin:
        log_line = fin.readlines()[-1]
        print("DEBUG:", log_line)
        assert log_line.startswith("Best")
        log_line = log_line.replace("GFLOPS", "")
        system_gflops = log_line.split(":")[1].split("/")
        system_gflops = [float(e.strip()) for e in system_gflops]
        # tccg "dispatches" to appropriate technique so
        # we take the fastest to be the tccg measurement
        tccg_gflops = max(system_gflops)
        return tccg_gflops


def dump_summary(summary_log, output_path):
    # tccg only does GEMM-like ops
    defaults = (
        "tccg",
        "+",
        "*",
    )

    with open(output_path, "w") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow([
            "system", "plus_op", "multiplies_op", "size", "isa", "benchmark",
            "gflops"
        ])
        for entry in summary_log:
            writer.writerow(defaults + entry)


def run_experiments(tccg_root_dir, sizes, archs, output_dir):
    summary_log = []

    for size in sizes:
        input_file = generate_cfg_file(size)
        for arch in archs:
            output_path = os.path.join(output_dir,
                                       "mm-{}-{}.txt".format(size, arch))
            print("Running", output_path)
            ret = run_tccg(tccg_root_dir, arch, input_file.name, output_path)
            if ret != 0:
                print("TCCG failed on", output_path)
            benchmark_name = "(+)(*){}".format(size)
            summary_log.append(
                (size, arch, benchmark_name, extract_gflops(output_path)))
        input_file.close()

    dump_summary(summary_log, os.path.join(output_dir, "summary.csv"))


def get_args():
    parser = ArgumentParser(description="Run MM benchmarks for TCCG")
    parser.add_argument("--tccg", type=str, help="Path to tccg root directory")
    parser.add_argument("--size", type=int, nargs="+", help="Sizes")
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
    run_experiments(args.tccg, args.size, args.arch, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
