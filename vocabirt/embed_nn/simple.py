from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import AUROC, Accuracy
from torch import nn

from .repr import Repr
from .utils import PrepareArgsMixin


class SimplePredictor(PrepareArgsMixin, pl.LightningModule):
    def __init__(
        self,
        *,
        vectors,
        num_candidates,
        lr=1e-3,
        C=1,
        feature_size=10,
        optimizer="adam",
        **kwargs,
    ):
        super().__init__()
        # feature_size * (2 if add_uncertainty else 1)
        self.trunk = Repr(vectors, feature_size, **kwargs)
        self.cls_heads = nn.ModuleList(
            [
                torch.nn.Linear(self.trunk.out_feature_size, 1)
                for _ in range(num_candidates)
            ]
        )
        self.lr = lr
        self.C = C
        self.optimizer = optimizer
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.auroc = AUROC(pos_label=1)
        self.acc = Accuracy()
        self.save_hyperparameters(
            *self.trunk.hparams, "lr", "C", "feature_size"  # , "add_uncertainty"
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--optimizer", default="adam")
        parser.add_argument("--C", type=float, default=1)
        parser.add_argument("--feature-size", type=int, default=10)
        # parser.add_argument(
        #    "--add-uncertainty", dest="add_uncertainty", action="store_true"
        # )
        # parser.add_argument(
        #    "--no-add-uncertainty", dest="add_uncertainty", action="store_false"
        # )
        # parser.set_defaults(add_uncertainty=False)
        return Repr.add_model_specific_args(parser)

    def forward_trunk(self, x: torch.Tensor):
        return self.trunk(x)

    def forward(self, resp_idx, x):
        feat = self.forward_trunk(x)
        return self.cls_heads[resp_idx](feat)[:, 0]

    def reg_loss(self, logits, ground_truth, head):
        model_loss = self.loss(logits, ground_truth)
        # Don't regularize bias to match scikit-learn/cuml
        l2_reg = 0.5 * torch.mm(head.weight, head.weight.t())
        return model_loss + l2_reg / self.C

    def training_step(self, batch, batch_idx):
        resp_idx, x, ground_truth = batch
        head = self.cls_heads[resp_idx]
        logits = self.forward(resp_idx, x)
        reg_loss = self.reg_loss(logits, ground_truth.float(), head)
        self.log("train_loss", reg_loss)
        return reg_loss

    def configure_optimizers(self):
        from .utils import get_optimizer

        return get_optimizer(self.optimizer, self.parameters(), self.lr)
