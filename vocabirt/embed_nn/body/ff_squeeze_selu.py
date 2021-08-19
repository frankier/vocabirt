from more_itertools import pairwise
from torch import nn

from .base import FFSqueezeBase


def ssn_init(lin):
    """
    Since SNNs have a fixed point at zero mean and unit variance for normalized
    weights ..., we initialize SNNs such that these constraintsare fulfilled in
    expectation. We draw the weights from a Gaussian distribution with E(wi) =
    0 and Var(wi) = 1/n. Uniform and truncated Gaussian distributions with
    these moments led to networks with similar behavior.
    """
    nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
    nn.init.zeros_(lin.bias)


class FFSeluDropoutSqueeze(FFSqueezeBase):
    def __init__(
        self,
        embedding_dim,
        *,
        input_dropout=0.2,
        hidden_dropout=0.2,
        hidden_layer_sizes=(300, 300, 300,),
        feature_size=10,
        **kwargs,
    ):
        super().__init__()
        self.input_dropout = (
            nn.Dropout(input_dropout) if input_dropout > 0 else nn.Sequential()
        )
        self.trunk = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.SELU(),
                    *([nn.Dropout(hidden_dropout)] if hidden_dropout > 0 else []),
                )
                for input_size, output_size in pairwise(
                    [embedding_dim, *hidden_layer_sizes]
                )
            )
        )
        self.output_linear = nn.Linear(
            hidden_layer_sizes[-1] if hidden_layer_sizes else embedding_dim,
            feature_size,
        )
        self.init_weights()

    def init_weights(self):
        for seq in self.trunk:
            ssn_init(seq[0])
        ssn_init(self.output_linear)
