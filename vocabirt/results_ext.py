import click

from .convergence_plots import load_scores


@click.command()
@click.argument("scores")
def main(scores):
    df = load_scores(scores, is_ext=True, load_last=True)
    final_results = df.groupby(["discrim_src", "diff_src", "discrim"])[
        ["auroc", "mcc", "pos_ap", "neg_ap"]
    ].agg("mean")
    print(final_results.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    main()
