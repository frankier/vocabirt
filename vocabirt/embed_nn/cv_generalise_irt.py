import pickle
from os import makedirs
from os.path import join as pjoin

import click
import numpy
import pandas
import torch

from embed_nn.cv_discrim import apply_stoi_list, train
from embed_nn.utils import get_numberbatch_vec, get_sv12k_word_list
from vocabirt.vocabirt2pl_svl12k_cv import split_modewise


def save_df(path, svl12k_words, words, difficulty, discrimination):
    word2cvidx = {w: i for i, w in enumerate(words)}
    reindex_cv2svl = numpy.asarray([word2cvidx[w] for w in svl12k_words])
    with open(path, "wb") as pickle_out:
        pickle.dump(
            {
                "words": pandas.DataFrame(
                    {
                        "word": svl12k_words,
                        "discrimination": torch.hstack(discrimination).numpy()[
                            reindex_cv2svl
                        ],
                        "difficulty": torch.hstack(difficulty).numpy()[reindex_cv2svl],
                    }
                )
            },
            pickle_out,
        )


@click.command()
@click.argument("resp_in", type=click.File("rb"))
@click.argument("irt_est_in")
@click.argument("outdir")
@click.option(
    "--split-mode", type=click.Choice(["word", "both", "none"]), default="none"
)
def main(resp_in, irt_est_in, outdir, split_mode):
    vectors = get_numberbatch_vec()
    resp_df = pandas.read_parquet(resp_in)
    svl12k_words = get_sv12k_word_list(resp_in)
    oov_words = {}
    best_models = {}
    train_outdir = pjoin(outdir, "train")
    makedirs(train_outdir, exist_ok=True)
    for resp_split_idx, vocab_split_idx, split_tpl, basename in split_modewise(
        split_mode, resp_df, svl12k_words
    ):
        path = pjoin(irt_est_in, f"{basename}.pkl")
        with open(path, "rb") as pickle_in:
            irt_est = pickle.load(pickle_in)
        best_models[(resp_split_idx, vocab_split_idx)] = train(
            ("difficulty", "discrimination"),
            vectors,
            irt_est["words"],
            split_tpl.train_words.to_numpy(),
            split_tpl.train_strata,
            outdir,
            resp_split_idx=resp_split_idx,
            vocab_split_idx=vocab_split_idx,
        )
        oov_words[vocab_split_idx] = split_tpl.test_words
    if split_mode != "none":
        results = {}
        for (
            (resp_split_idx, vocab_split_idx),
            (best_loss, best_model),
        ) in best_models.items():
            words = oov_words[vocab_split_idx]
            word_idxs = apply_stoi_list(vectors.stoi, words)
            preds = best_model(word_idxs)
            preds = preds.detach()
            if resp_split_idx not in results:
                resp_split_result = ([], [], [])
                results[resp_split_idx] = resp_split_result
            else:
                resp_split_result = results[resp_split_idx]
            resp_split_result[0].extend(words)
            resp_split_result[1].append(preds[:, 0])
            resp_split_result[2].append(preds[:, 1])
        pred_outdir = pjoin(outdir, "pred")
        makedirs(pred_outdir, exist_ok=True)
        for resp_split_idx, (words, difficulty, discrimination) in results.items():
            path = pjoin(pred_outdir, f"resp{resp_split_idx}.pkl")
            save_df(path, svl12k_words, words, difficulty, discrimination)


if __name__ == "__main__":
    main()
