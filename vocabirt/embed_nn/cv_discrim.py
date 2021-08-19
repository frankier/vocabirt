import pickle
from copy import deepcopy
from os import symlink, unlink
from os.path import join as pjoin
from os.path import lexists

import click
import numpy
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from torch import nn

from embed_nn.loader.common import apply_stoi
from vocabmodel.utils.freq import get_frequency_strata
from vocabmodel.utils.io_utils import read_word_list

from .repr import Repr
from .utils import get_numberbatch_vec, get_optimizer


def df_to_tensors(stoi, df, out_cols=("difficulty", "discrimination"), device="cpu"):
    words = torch.LongTensor(
        [apply_stoi(stoi, word) for word in df["word"]], device=device
    )
    out = torch.stack(
        [
            torch.FloatTensor(df[out_col].to_numpy(), device=device)
            for out_col in out_cols
        ],
        dim=-1,
    )
    return words, out


class Predictor(nn.Module):
    def __init__(
        self, *, vectors, feature_size=1, **kwargs,
    ):
        super().__init__()
        self.trunk = Repr(
            vectors,
            add_frequencies_input=False,
            add_frequencies=False,
            body="gelu_full_batch_norm",
            feature_size=feature_size,
            norm_output=False,
        )
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        return self.trunk(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss


def word_based_train_val_split(df, train_val_words, strata):
    train_val_split = StratifiedKFold(12, shuffle=True, random_state=42).split(
        train_val_words, strata
    )
    train_words, val_words = next(train_val_split)
    train_words = train_val_words[train_words]
    val_words = train_val_words[val_words]
    train_df = df[df["word"].isin(train_words)]
    val_df = df[df["word"].isin(val_words)]
    return train_df, val_df


def train(regressand, vectors, df, train_val_words, strata, outdir, **kwargs):
    model = Predictor(vectors=vectors, feature_size=len(regressand))
    opt = get_optimizer("adam", model.parameters(), 0.003)
    train_df, val_df = word_based_train_val_split(df, train_val_words, strata)
    train_words, train_out = df_to_tensors(
        vectors.stoi, train_df, out_cols=regressand, device="cpu"
    )
    val_words, val_out = df_to_tensors(
        vectors.stoi, val_df, out_cols=regressand, device="cpu"
    )
    logger = TensorBoardLogger(save_dir=outdir, default_hp_metric=False)
    sum_write = logger.experiment
    best_loss = float("inf")
    best_model = None
    best_out_path = None

    for epoch in range(50):
        # Train
        model.train()
        opt.zero_grad()
        loss = model.training_step((train_words, train_out))
        loss.backward()
        opt.step()
        train_loss = loss.item()
        sum_write.add_scalar("train_loss", train_loss, epoch)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = model.training_step((val_words, val_out)).item()
            sum_write.add_scalar("val_loss", val_loss, epoch)
        print("Epoch", epoch, train_loss, val_loss)

        # Save
        out_basename = f"epoch_{epoch}.tar"
        out_path = pjoin(outdir, out_basename)
        if best_model is None or val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model)
            best_out_path = out_basename
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_val_words": train_val_words,
                **kwargs,
            },
            out_path,
        )
    symlink_path = pjoin(outdir, "best.tar")
    if lexists(symlink_path):
        unlink(symlink_path)
    symlink(best_out_path, symlink_path)
    return (best_loss, best_model)


def apply_stoi_list(stoi, word_list):
    return torch.LongTensor([apply_stoi(stoi, word) for word in word_list])


def iter_vocab_splits(df_words, strata, train_out, num_splits=3):
    vocab_split = StratifiedKFold(num_splits, shuffle=True, random_state=42).split(
        df_words, strata
    )
    for word_split in range(num_splits):
        print(f"# Vocab split: {word_split + 1}/{num_splits}")
        train_val_words, test_words = next(vocab_split)
        outdir = pjoin(train_out, f"vocab{word_split}")
        yield word_split, df_words[train_val_words].to_numpy(), strata[
            train_val_words
        ], outdir
    print("# Overall")
    yield "overall", df_words, strata, pjoin(train_out, "overall"),


def inference(vectors, df_words, split_used_words, best_models, word_list, pred_out):
    with torch.no_grad():
        all_preds = {}
        inference_words = read_word_list(word_list)
        oov_words = []
        iv_words = []
        for word in inference_words:
            if word in df_words:
                iv_words.append(word)
            else:
                oov_words.append(word)
        oov_preds = best_models[-1][1](apply_stoi_list(vectors.stoi, oov_words))
        all_preds.update(zip(oov_words, oov_preds))
        for used_words, best_model in zip(split_used_words[:-1], best_models[:-1]):
            oov_split_words = [word for word in iv_words if word not in used_words]
            oov_split_preds = best_model[1](
                apply_stoi_list(vectors.stoi, oov_split_words)
            )
            all_preds.update(zip(oov_split_words, oov_split_preds))
        preds = torch.zeros(len(inference_words))
        for idx, word in enumerate(inference_words):
            preds[idx] = all_preds[word]
        pickle.dump((inference_words, preds), pred_out)


@click.command()
@click.argument("discrim_path", type=click.File("rb"))
@click.argument("train_out")
@click.argument("word_list", type=click.File("r"))
@click.argument("pred_out", type=click.File("wb"))
@click.option("--regressand", default=("difficulty", "discrimination"), multiple=True)
def main(discrim_path, train_out, word_list, pred_out, regressand):
    vectors = get_numberbatch_vec()
    df = pickle.load(discrim_path)
    df_words = df["word"]
    strata = numpy.asarray(get_frequency_strata(df_words, num_strata=5))
    # Splits
    split_used_words = []
    best_models = []
    for word_split, train_val_words, train_val_strata, outdir in iter_vocab_splits(
        df_words, strata, train_out
    ):
        print(f"# Vocab split: {word_split + 1}/3")
        split_used_words.append(train_val_words)
        best_models.append(
            train(
                regressand,
                vectors,
                df,
                df_words[train_val_words].to_numpy(),
                strata[train_val_words],
                outdir,
                word_split=word_split,
            )
        )
    # Inference
    inference(vectors, df_words, split_used_words, best_models, word_list, pred_out)


if __name__ == "__main__":
    main()
