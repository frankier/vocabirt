from argparse import ArgumentParser

import torch
from torch import nn

from .body.ff_squeeze_full_batch_norm import FFFullBatchNormDropoutSqueeze
from .body.ff_squeeze_selu import FFSeluDropoutSqueeze
from .features import get_frequency_embedding


def get_body_class(name):
    if name == "gelu_full_batch_norm":
        return FFFullBatchNormDropoutSqueeze
    else:
        assert name == "ssn"
        return FFSeluDropoutSqueeze


def add_frequencies_arg(parser):
    parser.add_argument(
        "--add-frequencies-input", dest="add_frequencies_input", action="store_true"
    )
    parser.add_argument(
        "--no-add-frequencies-input", dest="add_frequencies_input", action="store_false"
    )
    parser.add_argument(
        "--add-frequencies", dest="add_frequencies", action="store_true"
    )
    parser.add_argument(
        "--no-add-frequencies", dest="add_frequencies", action="store_false"
    )
    parser.set_defaults(add_frequencies_input=False, add_frequencies=False)


class Repr(nn.Module):
    def __init__(
        self,
        vectors,
        feature_size,
        *,
        add_frequencies_input,
        add_frequencies,
        body,
        **kwargs,
    ):
        super(Repr, self).__init__()
        self.vectors = vectors
        self._embed = None
        self._freq_embed = None
        embedding_dim = vectors.vectors.shape[1]
        if add_frequencies_input:
            embedding_dim += 1
        cls = get_body_class(body)
        self.inner = cls(embedding_dim, feature_size, **kwargs)
        self.add_frequencies_input = add_frequencies_input
        self.add_frequencies = add_frequencies
        if add_frequencies:
            self.freq_act = nn.Sequential(nn.Linear(1, 1), nn.Tanh())
            self.out_feature_size = feature_size
        else:
            self.out_feature_size = feature_size
        self.hparams = [
            *self.inner.hparams,
            "add_frequencies_input",
            "add_frequencies",
            "body",
        ]

    def embed(self, x):
        if self._embed is None:
            # Hide the embedding from PyTorch using a list so it doesn't get saved
            vectors_device = self.vectors.vectors.to(x.device)
            self._embed = [nn.Embedding.from_pretrained(vectors_device)]
        return self._embed[0](x)

    def freq_embed(self, x):
        if self._freq_embed is None:
            self._freq_embed = [
                get_frequency_embedding(self.vectors.stoi, "wordfreq", device=x.device)
            ]
        return self._freq_embed[0](x)

    def forward(self, x: torch.Tensor):
        embed_out = self.embed(x)
        if self.add_frequencies_input:
            embed_out = torch.hstack([embed_out, self.freq_embed(x)])
        if self.add_frequencies:
            return torch.hstack(
                [self.inner(embed_out), self.freq_act(self.freq_embed(x)),]
            )
        else:
            return self.inner(embed_out)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        add_frequencies_arg(parser)

        parser.add_argument(
            "--body",
            choices=["gelu_full_batch_norm", "ssn"],
            default="gelu_full_batch_norm",
        )
        temp_args, _ = parser.parse_known_args()
        body_cls = get_body_class(temp_args.body)
        return body_cls.add_model_specific_args(parser)

    def __deepcopy__(self, memo):
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "vectors":
                new_v = v
            else:
                new_v = deepcopy(v, memo)
            setattr(result, k, new_v)
        return result
