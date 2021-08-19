from more_itertools import pairwise
from torch import nn

from .base import FFSqueezeBase


class FFFullBatchNormDropoutSqueeze(FFSqueezeBase):
    def __init__(
        self,
        embedding_dim,
        feature_size,
        *,
        input_dropout=0.2,
        hidden_dropout=0.2,
        hidden_layer_sizes=(300, 300, 300,),
        norm_output=True,
        **kwargs,
    ):
        super().__init__()
        # Ordering of dropout, linear, relu based on analysis from:
        #  1. Understanding the Disharmony between Dropout and Batch
        #     Normalization by Variance Shift, Li et al. 2018
        #  2. Rethinking the Usage of Batch Normalization and Dropout in the
        #     Training of Deep Neural Networks, Chen et al. 2019

        # Don't put anything after feature output
        #  Why?
        #    - Because it is so low dimensional it it not likely to need
        #     dropout. Low dimentional is a form of regularisation in itself
        #    - We don't need reLu because the final linear layer is free to
        #     zero anything and add its own bias
        #    - We don't need batch norm because... (or do we? - just go without for now)

        # Okay now I added a batch norm without an on the output on the basis that
        # features for the linear regression should typically be scaled to be
        # able to meaningfully compare different values of C

        # Batch norm has momentum set to 1.0 so it always remembers the
        # mean/variance from the previous batch. At eval time we will use the
        # stats from the last batch. This is fine currently since we only ever
        # train on a whole batch at a time.

        # Might be worth investigating switching LayerNorm (at least at the
        # output) or adaptive gradient clipping if this assumption changes.
        self.input_dropout = (
            nn.Dropout(input_dropout) if input_dropout > 0 else nn.Sequential()
        )
        self.trunk = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.GELU(),
                    nn.BatchNorm1d(output_size, momentum=1.0),
                    *([nn.Dropout(hidden_dropout)] if hidden_dropout > 0 else []),
                )
                for input_size, output_size in pairwise(
                    [embedding_dim, *hidden_layer_sizes]
                )
            )
        )
        self.output_linear = nn.Sequential(
            nn.Linear(
                hidden_layer_sizes[-1] if hidden_layer_sizes else embedding_dim,
                feature_size,
            ),
            *(
                [nn.BatchNorm1d(feature_size, affine=False, momentum=1.0)]
                if norm_output
                else []
            ),
        )
        self.init_weights()

    def init_weights(self):
        # TODO: LSUV as in GELU paper
        # https://gist.github.com/simongrest/52404966f0c46f750a823a44618bb06c
        # https://github.com/ducha-aiki/LSUV-pytorch
        # https://github.com/shunk031/LSUV.pytorch
        pass
