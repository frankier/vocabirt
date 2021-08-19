import os
import pickle
from os.path import join as pjoin

import click
import numpy


@click.command()
@click.argument("gold", type=click.File("rb"))
@click.argument("est")
def main(gold, est):
    gold_df = pickle.load(gold)["words"]
    diffs = []
    discrim = []
    for name in os.listdir(est):
        if not name.endswith(".pkl"):
            continue
        df = pickle.load(open(pjoin(est, name), "rb"))["words"]
        diffs.append(df["difficulty"] - gold_df["difficulty"])
        discrim.append(df["discrimination"] - gold_df["discrimination"])
    diffs = numpy.vstack(diffs)
    print("Difficulty")
    abs_diffs = numpy.abs(diffs)
    print("MAE {:.3f} {:.3f}".format(abs_diffs.mean(), abs_diffs.std()))
    print("MSE {:.3f}".format((diffs ** 2).mean()))
    print("Norm MAE {:.3f}".format(abs_diffs.mean() / gold_df["difficulty"].std()))
    discrim = numpy.vstack(discrim)
    print("Discriminations")
    abs_discrim = numpy.abs(discrim)
    print("MAE {:.3f} {:.3f}".format(abs_discrim.mean(), abs_discrim.std()))
    print("MSE {:.3f}".format((discrim ** 2).mean()))
    print(
        "Norm MAE {:.3f}".format(abs_discrim.mean() / gold_df["discrimination"].std())
    )


if __name__ == "__main__":
    main()
