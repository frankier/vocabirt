import math
import os

import pandas
import torch
from cachetools import cached
from torch import nn

from embed_nn.loader.common import apply_stoi
from embed_nn.utils import get_sv12k_word_list
from vocabmodel.utils.freq import MAX_CBPACK, get_word_buckets


def get_word_prevalences():
    word_prevalence_path = os.environ["WORD_PREVALENCE"]
    df = pandas.read_csv(word_prevalence_path)
    for idx, row in df.iterrows():
        yield row["Word"], row["Prevalence"]


def get_word_aoa():
    kuperman_aoa = os.environ["KUPERMAN_AOA"]
    df = pandas.read_csv(kuperman_aoa)
    for idx, row in df.iterrows():
        if math.isnan(row["Rating.Mean"]):
            continue
        yield row["Word"], row["Rating.Mean"]


@cached(cache={}, key=lambda stoi, device=None,: (id(stoi), device))
def get_frequency_embedding(stoi, device=None):
    max_stoi = max(stoi.values())
    embedding_vecs = torch.full(
        (max_stoi + 1, 1), MAX_CBPACK + 50, dtype=torch.float, device=device
    )
    for word, idx in get_word_buckets():
        if word not in stoi:
            continue
        embedding_vecs[stoi[word]] = idx
    return nn.Embedding.from_pretrained(embedding_vecs)


def get_norm_frequency_embedding(stoi, words=None, vocab_response_path=None):
    embed = get_frequency_embedding(stoi)
    if words is None:
        words = get_sv12k_word_list(vocab_response_path)
        words = torch.as_tensor([apply_stoi(stoi, word) for word in words])
    train_embedded = embed(words)
    embed.weight[:] = (embed.weight - torch.mean(train_embedded, 0)) / torch.std(
        train_embedded, 0
    )
    return embed


def get_prevalence_embedding(stoi, device=None):
    max_stoi = max(stoi.values())
    embedding_vecs = torch.full(
        (max_stoi + 1, 1), -10, dtype=torch.float, device=device
    )
    for word, prevalence in get_word_prevalences():
        if word not in stoi:
            continue
        embedding_vecs[stoi[word]] = prevalence
    return nn.Embedding.from_pretrained(embedding_vecs)


def get_aoa_embedding(stoi, device=None):
    max_stoi = max(stoi.values())
    embedding_vecs = torch.full((max_stoi + 1, 1), 50, dtype=torch.float, device=device)
    for word, aoa in get_word_aoa():
        if word not in stoi:
            continue
        embedding_vecs[stoi[word]] = aoa
    return nn.Embedding.from_pretrained(embedding_vecs)
