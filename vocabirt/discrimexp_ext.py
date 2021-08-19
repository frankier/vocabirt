import pickle

import click
import numpy
import pandas
import wordfreq
from catsim.estimation import HillClimbingEstimator

from .discrimexp import Scorer, print_summary
from .estimators import LogisticEstimator


def load_bank(irt_df, words):
    from catsim.irt import normalize_item_bank

    # from .vocabirt2pl_svl12k import get_word_difficulties
    # difficulties = get_word_difficulties(words, "wordfreq_norm")
    irt_words = list(irt_df["word"])
    # perm = [(irt_words.index(word) if word in irt_words else -1) for word in words]
    perm = [irt_words.index(word) for word in words]
    bank = numpy.hstack(
        [
            irt_df["discrimination"].to_numpy()[perm, numpy.newaxis],
            irt_df["difficulty"].to_numpy()[perm, numpy.newaxis],
        ]
    )
    # for idx, (val, word) in enumerate(zip(perm, words)):
    #    if val != -1:
    #        continue
    #    bank[idx, 0] = 1.2
    #    bank[idx, 1] = difficulties[idx]
    return normalize_item_bank(bank)


def comb_indices(words, num_items):
    freqs = [
        (wordfreq.word_frequency(word, "en"), idx) for idx, word in enumerate(words)
    ]
    freqs.sort()
    return [
        freqs[int((len(freqs) - 1) * float(idx) / num_items + 0.5)][1]
        for idx in range(num_items)
    ]


@click.command()
@click.argument("vocab_response_path")
@click.argument("irt_path", type=click.File("rb"))
@click.argument("outf", type=click.File("wb"))
@click.option("--fmt", type=click.Choice(["testyourvocab", "evkd1"]))
@click.option("--no-discrim-preds/--discrim-preds")
@click.option("--verbose/--terse")
def main(vocab_response_path, irt_path, outf, fmt, no_discrim_preds, verbose):
    scorer = Scorer()
    if fmt == "evkd1":
        estimator = HillClimbingEstimator(dodd=True)
    else:
        estimator = LogisticEstimator(use_discriminations=False)
    irt_df = pickle.load(irt_path)["words"]
    resp_df = pandas.read_parquet(vocab_response_path)
    if fmt == "evkd1":
        resp_df["known"] = resp_df["correct"]
    resp_id_col = "user_id" if fmt == "testyourvocab" else "resp_id"
    with click.progressbar(resp_df.groupby(resp_id_col)) as user_resp:
        for user_id, resp in user_resp:
            known = resp["known"]
            if known.all() or not known.any():
                continue
            administered_items = comb_indices(resp["word"], 40)
            bank = load_bank(irt_df, resp["word"])
            if fmt == "evkd1":
                bank[:, 2] = 0.25
            no_discrim_bank = bank.copy()
            no_discrim_bank[:, 0] = 1
            if no_discrim_preds:
                icc_bank = no_discrim_bank
            else:
                icc_bank = bank

            responses = list(known.iloc[administered_items])
            theta = estimator.estimate(
                items=no_discrim_bank,
                administered_items=administered_items,
                response_vector=responses,
                est_theta=0,
            )
            scorer.add_scores(user_id, 40, icc_bank, theta, known.to_numpy())
    last_df = scorer.as_df(last_only=True)
    print_summary(last_df)
    pickle.dump(
        {"last_df": last_df,}, outf,
    )


if __name__ == "__main__":
    main()
