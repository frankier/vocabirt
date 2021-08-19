from argparse import ArgumentParser

from torch import nn


class FFSqueezeBase(nn.Module):
    hparams = [
        "input_dropout",
        "hidden_dropout",
        "hidden_layer_sizes",
        "feature_size",
    ]

    def forward(self, x):
        return self.output_linear(self.trunk(self.input_dropout(x)))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input-dropout", type=float, default=0.2)
        parser.add_argument("--hidden-dropout", type=float, default=0.2)
        parser.add_argument(
            "--hidden-layer-sizes", nargs="+", type=int, default=[300, 300, 300]
        )
        parser.add_argument(
            "--linear", dest="hidden_layer_sizes", action="store_const", const=[]
        )
        return parser
