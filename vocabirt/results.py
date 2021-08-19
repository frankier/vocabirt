import click
import pandas as pd

from .convergence_plots import load_scores


def load_scores_errs(scores, abilities):
    df = load_scores(scores, load_last=True)
    df.reset_index(inplace=True)
    true_abilities = pd.Series([abilities[idx - 1] for idx in df["respondent"]])
    df["theta_err"] = df["theta"] - true_abilities
    df["abs_theta_err"] = df["theta_err"].abs()
    return df


@click.command()
@click.argument("scores")
def main(scores):
    df = load_scores(scores)
    df = df[df["strategy"] == "urry"]
    final_results = df.groupby(["split_mode", "diff", "discrim"])[
        ["auroc", "mcc", "pos_ap", "neg_ap"]
    ].agg("mean")
    print(final_results.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    main()
