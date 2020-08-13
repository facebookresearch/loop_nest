#!/usr/bin/env python3
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def get_barplot(df, field, ylabel=None, title=None):
    ylabel = field.upper() if ylabel is None else ylabel
    _, ax = plt.subplots(1)
    sns.barplot(ax=ax, data=df, x="benchmark", y=field, hue="system")
    ax.set_xlabel("Benchmark")
    ax.set_ylabel(ylabel)
    ax.get_legend().set_title("System")
    if title is not None:
        ax.set_title(title)
    return ax


def get_ratio_plot(df,
                   denominator,
                   field,
                   ylabel=None,
                   title=None,
                   add_unit_line=False):
    df = df.copy()

    is_denom = df["system"] == denominator
    reference_df = df[is_denom][["benchmark", field]]
    df = df[~is_denom]

    reference_df = reference_df.rename(columns={field: "reference_value"})
    reference_df = reference_df.reset_index(drop=True)
    df = pd.merge(df, reference_df, how="left", on="benchmark")
    df["ratio"] = df[field] / df["reference_value"]
    df["system"] = df["system"].map(lambda x: x + "/" + denominator)
    plot = get_barplot(df, "ratio", ylabel, title)

    if add_unit_line:
        xmin, xmax, _, _ = plot.axis()
        plot.hlines(y=1.0, xmin=xmin, xmax=xmax, linestyle='--', color='r')

    return plot


def get_args():
    parser = ArgumentParser(description="Plot benchmark values")
    parser.add_argument(
        "--input",
        nargs="+",
        type=str,
        help="Input summary csv files",
    )
    parser.add_argument(
        "--field",
        type=str,
        help="Field to plot",
    )
    parser.add_argument(
        "--filter_field",
        type=str,
        nargs="+",
        help="Fields to filter on",
    )
    parser.add_argument(
        "--filter_value",
        type=str,
        nargs="+",
        help="Field values to filter on",
    )
    parser.add_argument(
        "--denominator",
        type=str,
        help="System to use as denominator if ratio",
    )
    parser.add_argument(
        "--add_unit_line",
        action="store_true",
        help="Add horizontal line at 1.0",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        help="Ylabel",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title",
    )
    parser.add_argument("--output", type=str, help="Output pdf file")
    return parser.parse_args()


def main():
    args = get_args()

    dfs = []
    for input_path in args.input:
        df = pd.read_csv(input_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)

    # drop any row with missing value (i.e. we didn't measure it)
    combined_df = combined_df[~pd.isnull(combined_df[args.field])]

    if args.filter_field:
        for f, v in zip(args.filter_field, args.filter_value):
            combined_df = combined_df[combined_df[f].astype(str) == v]

    isas = combined_df.isa.unique()
    plots = []

    for isa in isas:
        title = args.title + "({})".format(
            isa.upper()) if args.title else isa.upper()
        isa_df = combined_df[combined_df["isa"] == isa]
        if args.denominator:
            plot = get_ratio_plot(
                isa_df,
                denominator=args.denominator,
                field=args.field,
                ylabel=args.ylabel,
                title=title,
                add_unit_line=args.add_unit_line,
            )
        else:
            plot = get_barplot(isa_df,
                               field=args.field,
                               ylabel=args.ylabel,
                               title=title)

        plots.append(plot)

    with PdfPages(args.output) as pdf_out:
        for p in plots:
            pdf_out.savefig(p.get_figure())


if __name__ == "__main__":
    try:
        main()
    except Exception as err: # noqa F841
        import pdb

        pdb.post_mortem()
