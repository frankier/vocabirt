from os import makedirs
from os.path import join as pjoin

import click
import numpy
import pandas
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from embed_nn.loader.common import SVL12KDataset
from embed_nn.simple import SimplePredictor
from vocabmodel.utils.freq import add_freq_strata, get_frequency_strata

from .cv import freq_init_weights, get_ds_data, train_batch
from .cv_discrim import inference, iter_vocab_splits, word_based_train_val_split
from .utils import get_numberbatch_vec, get_sv12k_word_list


def train(vectors, outdir, model, df, words, strata, multiple_thresh=False, **kwargs):
    train_df, val_df = word_based_train_val_split(df, words, strata)
    logger = TensorBoardLogger(save_dir=outdir, default_hp_metric=False)
    train_words, train_strata, all_train_knows = get_ds_data(
        SVL12KDataset(vectors.stoi, train_df, multiple_thresh=multiple_thresh),
        model.device,
    )[:3]
    val_words, val_strata, all_val_knows = get_ds_data(
        SVL12KDataset(vectors.stoi, val_df, multiple_thresh=multiple_thresh),
        model.device,
    )[:3]
    freq_init_weights(vectors.stoi, model, train_words, all_train_knows)

    opt = model.configure_optimizers()
    makedirs(outdir, exist_ok=True)
    sum_write = logger.experiment
    best_loss = float("inf")
    best_model = None

    for epoch in range(10):
        # Train
        model.train()
        train_loss = train_batch(opt, model, train_words, all_train_knows)
        sum_write.add_scalar("train_loss", train_loss, epoch)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, val_knows in enumerate(all_val_knows):
                loss = model.training_step((batch_idx, val_words, val_knows), batch_idx)
                val_loss += loss.item()
            sum_write.add_scalar("val_loss", val_loss, epoch)
        print("Epoch", epoch, train_loss, val_loss)

        # Save
        if best_model is None or val_loss < best_loss:
            best_loss = val_loss
            best_model = model
        out_path = pjoin(outdir, f"epoch_{epoch}.tar")
        torch.save(
            {
                "epoch": epoch,
                "hyper_parameters": dict(model.hparams),
                "state_dict": model.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_val_words": words,
                **kwargs,
            },
            out_path,
        )
    return (best_loss, best_model)


@click.command()
@click.argument("vocab_response_path")
@click.argument("train_out")
@click.argument("word_list", type=click.File("r"))
@click.argument("pred_out", type=click.File("wb"))
def main(vocab_response_path, train_out, word_list, pred_out):
    vectors = get_numberbatch_vec()
    words = get_sv12k_word_list(vocab_response_path)
    df = pandas.read_parquet(vocab_response_path)
    add_freq_strata(df)
    df = df[df["respondent"] != 0]
    strata = numpy.asarray(get_frequency_strata(words, num_strata=5))
    # Splits
    split_used_words = []
    best_models = []
    for word_split, train_val_words, train_val_strata, outdir in iter_vocab_splits(
        words, strata, train_out,
    ):
        train_val_df = df[df["word"].isin(train_val_words)]
        split_used_words.append(train_val_words)
        args = SimplePredictor.prepare_args(optimizer="adam", lr=0.003, C=1000)
        model = SimplePredictor(**args, vectors=vectors, num_candidates=15)
        best_models.append(
            train(
                vectors,
                outdir,
                model,
                train_val_df,
                word_split,
                train_val_words,
                train_val_strata,
                word_split=word_split,
            )
        )
        best_models_trunk = [
            (loss, lambda x: model.trunk(x)[:, 0]) for loss, model in best_models
        ]
    inference(vectors, words, split_used_words, best_models_trunk, word_list, pred_out)


if __name__ == "__main__":
    main()
