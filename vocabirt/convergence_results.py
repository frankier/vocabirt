import pickle

import click
import numpy as np
import pandas as pd

from .results import load_scores_errs


@click.command()
@click.argument("scores")
@click.argument("abilities", type=click.File("rb"))
def main(scores, abilities):
    abilities = pickle.load(abilities)["abilities"]
    abilities_std = pd.Series(abilities).std()
    df = load_scores_errs(scores, abilities)
    df = df[(df["diff"] == "raw") & df["discrim"]]
    grouped = df.groupby(["split_mode", "estimator", "strategy"])
    agg = grouped.agg(
        {"abs_theta_err": [np.mean, lambda x: np.mean(x) / abilities_std]}
    )
    print(agg.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    main()
