import pickle
from os.path import join as pjoin

import click
import pandas

from vocabirt.embed_nn.loader.common import CVSplitEhara
from vocabirt.embed_nn.utils import get_sv12k_word_list

from .vocabirt2pl_svl12k import estimate_irt


def split_modewise(split_mode, df, words):
    total_resp_splits = 3 if split_mode in ("respondent", "both") else 1
    total_vocab_splits = 3 if split_mode in ("word", "both") else 1
    for resp_split_idx in range(total_resp_splits):
        for vocab_split_idx in range(total_vocab_splits):
            splitter = CVSplitEhara(
                words,
                resp_split_idx=resp_split_idx,
                vocab_split_idx=vocab_split_idx,
                total_resp_splits=total_resp_splits,
                total_vocab_splits=total_vocab_splits,
            )
            yield resp_split_idx, vocab_split_idx, splitter.split_all(
                df
            ), f"resp{resp_split_idx}_vocab{vocab_split_idx}"


split_mode_opt = click.option(
    "--split-mode", type=click.Choice(["none", "respondent", "word", "both"])
)


@click.command()
@click.argument("inf")
@click.argument("outdir")
@click.option("--difficulties")
@split_mode_opt
def main(inf, outdir, difficulties, split_mode):
    df = pandas.read_parquet(inf)
    words = get_sv12k_word_list(inf)
    for resp_split_idx, vocab_split_idx, split_tpl, basename in split_modewise(
        split_mode, df, words
    ):
        path = pjoin(outdir, f"{basename}.pkl")
        with open(path, "wb") as pickle_out:
            pickle.dump(
                estimate_irt(split_tpl.train_df, ordinal=True), pickle_out,
            )


if __name__ == "__main__":
    main()
