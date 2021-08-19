import os
from argparse import ArgumentParser
from os import makedirs
from os.path import join as pjoin

import numpy
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.linear_model import LogisticRegression

from embed_nn.loader.common import CVSplitEhara
from embed_nn.loader.cv import uniform_stratified_shuffle_split

from .features import (
    get_aoa_embedding,
    get_frequency_embedding,
    get_norm_frequency_embedding,
    get_prevalence_embedding,
)
from .utils import get_model_classes, get_numberbatch_vec, get_sv12k_word_list

DEFAULT_MAX_EPOCHS = 50


def get_ds_data(ds, device):
    words = None
    strata = None
    resp_data = []
    for resp_dataset in ds.resp_datasets:
        if words is None:
            words = torch.LongTensor(
                [word for word, _, _ in resp_dataset], device=device
            )
        if strata is None:
            strata = numpy.asarray([stratum for _, _, stratum in resp_dataset])
        resp_data.append(
            torch.BoolTensor([knows for _, knows, _ in resp_dataset], device=device),
        )
    return words, strata, torch.vstack(resp_data), torch.as_tensor(ds.respondent_ids)


def freq_init_weights(stoi, model, train_words, all_train_knows):
    embed = get_norm_frequency_embedding(stoi, train_words)
    logit = LogisticRegression(C=model.C, max_iter=300)
    for idx, knows in enumerate(all_train_knows):
        logit.fit(embed(train_words), knows)
        weight = logit.coef_
        bias = logit.intercept_
        feat_size = model.trunk.out_feature_size
        cls_head = model.cls_heads[idx]
        weight = torch.as_tensor(np.repeat(weight, feat_size, axis=1) / feat_size)
        bias = torch.as_tensor(bias)
        with torch.no_grad():
            cls_head.weight[:] = weight
            cls_head.bias[:] = bias


def train_batch(opt, model, train_words, all_train_knows):
    loss_sum = 0
    for batch_idx, train_knows in enumerate(all_train_knows):
        opt.zero_grad()
        loss = model.training_step((batch_idx, train_words, train_knows), batch_idx)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    return loss_sum
