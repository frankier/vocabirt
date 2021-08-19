import os
import pickle
from os import makedirs
from os.path import join as pjoin

import click

from .vocabirt2pl_svl12k import get_difficulties


def patch_difficulty(inf, outf):
    contents = pickle.load(inf)
    df = contents["words"]
    df["difficulty"] = (
        df["difficulty"].mean()
        + get_difficulties(df, "wordfreq_norm") * df["difficulty"].std()
    )
    pickle.dump(contents, outf)


@click.command()
@click.argument("indir")
@click.argument("outdir")
def main(indir, outdir):
    for root, dirs, files in os.walk(indir):
        for name in files:
            if not name.endswith(".pkl"):
                continue
            out_root = pjoin(outdir, root[len(indir) :])
            makedirs(out_root, exist_ok=True)
            with open(pjoin(root, name), "rb") as inf, open(
                pjoin(out_root, name), "wb"
            ) as outf:
                patch_difficulty(inf, outf)


if __name__ == "__main__":
    main()
