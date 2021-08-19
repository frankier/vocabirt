import os
from pathlib import Path

import torch
from torch.optim.lr_scheduler import MultiplicativeLR
import pandas
from torchtext.vocab import Vectors


def get_sv12k_word_list(vocab_response_path=None):
    if vocab_response_path is None:
        vocab_response_path = os.environ["VOCAB_RESPONSE_PATH"]
    df = pandas.read_parquet(vocab_response_path)
    return df[df["respondent"] == 0]["word"]


def get_numberbatch_vec(numberbatch_path=None):
    if numberbatch_path is None:
        numberbatch_path = os.environ["NUMBERBATCH_PATH"]
    numberbatch_path = Path(numberbatch_path)
    return Vectors(numberbatch_path.stem, cache=numberbatch_path.parent)


def get_model_classes(model_name: str):
    assert model_name == "simple"
    from embed_nn.loader.simple import SVL12KLoader as SimpleSVL12KLoader
    from embed_nn.simple import SimplePredictor

    return SimplePredictor, SimpleSVL12KLoader


class PrepareArgsMixin:
    @classmethod
    def prepare_args(cls, **kwargs):
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser = cls.add_model_specific_args(parser)
        default_args = parser.parse_args([])
        res = vars(default_args)
        res.update(**kwargs)
        return res


def get_optimizer(optimizer_name, parameters, lr):
    if optimizer_name in ("sgd", "sgddecayswa"):
        optim_fn = torch.optim.SGD
    elif optimizer_name in ("adamw", "adamwdecayswa"):
        optim_fn = torch.optim.AdamW
    else:
        optim_fn = torch.optim.Adam
    if optimizer_name in ("sgddecayswa", "adamwdecayswa", "adamdecayswa"):
        optimizer = optim_fn(parameters, lr=lr)
        scheduler = MultiplicativeLR(
            optimizer, lr_lambda=lambda epoch: 0.5 if (epoch % 8 == 7) else 1
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
    else:
        return optim_fn(parameters, lr=lr)
