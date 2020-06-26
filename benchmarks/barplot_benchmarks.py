#!/usr/bin/env python3
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def get_benchmark(line):
    m = re.match("Benchmark: ([0-9]+)", line)
    if m is not None:
        return int(m.group(1))


def get_maxabsdiff(line):
    m = re.match("MAXABSDIFF: ([^\n ]*$)", line)
    if m is not None:
        return float(m.group(1))


def get_gflops(line):
    m = re.match("GFLOPS: ([^\n ]*$)", line)
    if m is not None:
        return float(m.group(1))


def get_compile(line):
    m = re.match("Compile: ([^\n ]*$)", line)
    if m is not None:
        return float(m.group(1))


def df_from_log(log_path, label):
    tokenizers = {
        "benchmark": (get_benchmark, []),
        "compile": (get_compile, []),
        "diff": (get_maxabsdiff, []),
        "gflops": (get_gflops, []),
    }

    with open(log_path, "r") as fin:
        for line in fin:
            for token_func, values in tokenizers.values():
                result = token_func(line)
                if result is not None:
                    values.append(result)
                    break

    len_vecs = {len(v) for _, v in tokenizers.values()}
    assert len(len_vecs) == 1, "Mismatched lengths, check parse"

    df = pd.DataFrame({k: vs for k, (_, vs) in tokenizers.items()})
    df["label"] = label

    large_diffs = df["diff"] > 1e-4
    if large_diffs.any():
        print(
            "Diffs in {}; check outputs".format(
                ", ".join(df[large_diffs]["benchmark"].astype(str))
            )
        )
        raise Exception("Large differences in {} benchmarks".format(large_diffs.sum()))
    return df


def get_barplot(df, field="gflops", ylabel="GLOPS"):
    _, ax = plt.subplots(1)
    sns.barplot(ax=ax, data=df, x="benchmark", y=field, hue="label")
    ax.set_xlabel("Benchmark")
    ax.set_ylabel(ylabel)
    ax.get_legend().set_title("System")
    return ax


def get_ratio_plot(df, denominator="Halide", field="gflops", ylabel="GFLOPS"):
    df = df.copy()

    is_denom = df["label"] == denominator
    reference_df = df[is_denom][["benchmark", field]]
    df = df[~is_denom]

    reference_df = reference_df.rename(columns={field: "reference_value"})
    reference_df = reference_df.reset_index(drop=True)
    df = pd.merge(df, reference_df, how="left", on="benchmark")
    df["ratio"] = df[field] / df["reference_value"]
    df["label"] = df["label"].map(lambda x: x + "/" + denominator)
    return get_barplot(df, "ratio", ylabel)


def get_args():
    parser = ArgumentParser(description="Plot benchmark GLOPs")
    parser.add_argument(
        "--input",
        nargs="+",
        type=str,
        help="Input files (stdout from .cpp benchmark files)",
    )
    parser.add_argument(
        "--label",
        nargs="+",
        type=str,
        help="Labels for series (aligned with input files)",
    )
    parser.add_argument("--output", type=str, help="Output pdf file")
    return parser.parse_args()


def main():
    args = get_args()

    dfs = []
    for input_path, label in zip(args.input, args.label):
        input_df = df_from_log(input_path, label)
        dfs.append(input_df)
    combined_df = pd.concat(dfs, axis=0)

    gflops_plot = get_barplot(combined_df, field="gflops", ylabel="GFLOPS")
    compile_plot = get_barplot(
        combined_df, field="compile", ylabel="Compile or CodeGen Time (seconds)"
    )

    gflops_ratio_plot = get_ratio_plot(
        combined_df, denominator="Halide", field="gflops", ylabel="Ratio of GLOPS"
    )
    compile_ratio_plot = get_ratio_plot(
        combined_df,
        denominator="loop_nest",
        field="compile",
        ylabel="Ratio of Compile/CodeGen Time",
    )

    plots = [gflops_plot, gflops_ratio_plot, compile_plot, compile_ratio_plot]
    with PdfPages(args.output) as pdf_out:
        for p in plots:
            pdf_out.savefig(p.get_figure())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import pdb

        pdb.post_mortem()
