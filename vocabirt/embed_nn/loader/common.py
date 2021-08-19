import abc
import os
import random
import re
from collections import namedtuple
from itertools import islice, repeat

import pandas
import pytorch_lightning as pl
from loky import cpu_count
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import BatchSampler, DataLoader, Dataset

from embed_nn.utils import get_sv12k_word_list
from vocabmodel.utils.freq import add_freq_strata, get_frequency_strata


def get_cv_split_ehara(vocab_response_path, spec):
    words = get_sv12k_word_list(vocab_response_path)
    resp_split_idx, vocab_split_idx = [int(x) for x in spec.split("_")]
    return CVSplitEhara(
        words, resp_split_idx=resp_split_idx, vocab_split_idx=vocab_split_idx
    )


def incr_list(l):
    return [x + 1 for x in l]


def get_resp_split(num_respondents, resp_split_idx, total_resp_splits):
    resp_idxs = range(num_respondents)
    if total_resp_splits == 1:
        return incr_list(resp_idxs), []
    fold = KFold(total_resp_splits, shuffle=True, random_state=42).split(resp_idxs)
    train_resp, test_resp = next(islice(fold, resp_split_idx, resp_split_idx + 1))
    return incr_list(train_resp), incr_list(test_resp)


def get_vocab_split(words, strata, vocab_split_idx, total_vocab_splits):
    if total_vocab_splits == 1:
        return list(range(len(words))), []
    vocab_split = StratifiedKFold(
        total_vocab_splits, shuffle=True, random_state=42
    ).split(words, strata)
    return next(islice(vocab_split, vocab_split_idx, vocab_split_idx + 1))


SplitData = namedtuple(
    "SplitData",
    "train_df val_df test_df train_words test_words train_strata test_strata",
)


class CVSplitEhara:
    """
    CV split which leaves part of the vocabulary out
    """

    def __init__(
        self,
        words,
        num_respondents=15,
        resp_split_idx=0,
        vocab_split_idx=0,
        total_resp_splits=3,
        total_vocab_splits=3,
    ):
        self.words = words
        self.num_respondents = num_respondents
        self.resp_split_idx = resp_split_idx
        self.vocab_split_idx = vocab_split_idx
        self.total_resp_splits = total_resp_splits
        self.total_vocab_splits = total_vocab_splits

    def _split(self, df):
        df = df[df["respondent"] != 0]
        train_resp, test_resp = get_resp_split(
            self.num_respondents, self.resp_split_idx, self.total_resp_splits
        )
        strata = get_frequency_strata(self.words, num_strata=5)
        train_words, test_words = get_vocab_split(
            self.words, strata, self.vocab_split_idx, self.total_vocab_splits
        )
        train_strata = strata[train_words]
        test_strata = strata[test_words]
        train_words = self.words[train_words]
        test_words = self.words[test_words]
        train_resp_df = df[df["respondent"].isin(train_resp)]
        train_df = train_resp_df[train_resp_df["word"].isin(train_words)]
        val_df = train_resp_df[train_resp_df["word"].isin(test_words)]
        test_df = df[df["respondent"].isin(test_resp)]
        return SplitData(
            train_df,
            val_df,
            test_df,
            train_words,
            test_words,
            train_strata,
            test_strata,
        )

    def split_all(self, df):
        return self._split(df)

    def __call__(self, df):
        return self._split(df)[:3]

    def split4(self, df):
        train_df, val_df, test_df, train_words, test_words = self._split(df)
        return (
            train_df,
            val_df,
            test_df[test_df["word"].isin(train_words)],
            test_df[test_df["word"].isin(test_words)],
        )


def fixed_split_ehara(df):
    """
    This is the original split, which probably shouldn't be used going forward
    because it doesn't prevent remembering of word vectors.
    """
    add_freq_strata(df)
    good_responses = df["respondent"] != 0
    non_japanese = (df["respondent"] == 3) | (df["respondent"] == 14)
    test_df = df[non_japanese]
    rest_df = df[good_responses & ~non_japanese]
    strata = [f"{r}___{s}" for r, s in zip(rest_df["respondent"], rest_df["stratum"])]
    k_fold = StratifiedKFold(12, shuffle=True, random_state=42)
    train_idxs, val_idxs = next(k_fold.split(rest_df, strata))
    train_df = rest_df.iloc[train_idxs]
    val_df = rest_df.iloc[val_idxs]
    return train_df, val_df, test_df


class TrainValSplitEhara:
    """
    Based on the CV split but leaves the test partition empty, this is for
    usage testing on the out of domain data like the testyourvocab data.
    """

    def __init__(
        self, words,
    ):
        self.words = words
        strata = get_frequency_strata(words, num_strata=5)
        self.vocab_split = StratifiedKFold(12, shuffle=True, random_state=42).split(
            words, strata
        )

    def __call__(self, df):
        add_freq_strata(df)
        df = df[df["respondent"] != 0]
        train_words, test_words = next(self.vocab_split)
        train_words = self.words[train_words]
        test_words = self.words[test_words]
        train_df = df[df["word"].isin(train_words)]
        val_df = df[df["word"].isin(test_words)]
        return train_df, val_df, train_df[0:0]


class RespondentDataset(Dataset):
    def __init__(self, stoi, df, thresh=5):
        self.stoi = stoi
        self.df = df
        self.thresh = thresh

    def __getitem__(self, row_idx):
        row = self.df.iloc[row_idx]
        return (
            apply_stoi(self.stoi, row["word"]),
            row["score"] >= self.thresh,
            row["stratum"],
        )

    def __len__(self):
        return len(self.df)


class SVL12KDataset(Dataset):
    def __init__(self, stoi, df, multiple_thresh=False):
        groupby = df.groupby("respondent")
        self.resp_datasets = []
        self.respondent_ids = []
        for grp in groupby.groups:
            group_df = groupby.get_group(grp).reset_index(drop=True)
            if multiple_thresh:
                for thresh in range(2, 6):
                    knows = group_df["score"] >= thresh
                    if knows.all() or not knows.any():
                        continue
                    self.resp_datasets.append(
                        RespondentDataset(stoi, group_df, thresh=thresh)
                    )
            else:
                self.resp_datasets.append(RespondentDataset(stoi, group_df))
            self.respondent_ids.append(grp)
        if multiple_thresh:
            if len(self.respondent_ids) == 0:
                self.len = 0
            else:
                self.len = len(self.resp_datasets) * (
                    len(df) // len(self.respondent_ids)
                )
        else:
            self.len = len(df)

    def __getitem__(self, index):
        resp_idx, row_idx = index
        tpl = self.resp_datasets[resp_idx][row_idx]
        return (*tpl, self.respondent_ids[resp_idx])

    def __len__(self):
        return self.len


class RespondentPermSamplerBase(abc.ABC, BatchSampler):
    def __init__(self, dataset, generator=None, shuffle=True):
        self.dataset = dataset
        self.generator = generator
        self.random = random.Random()
        self.shuffle = shuffle
        self.resp_sampler = []
        for resp_dataset in self.dataset.resp_datasets:
            self.resp_sampler.append(self.get_inner_sampler(resp_dataset))
        self.resp_perm = []
        for resp_id, batch_sampler in enumerate(self.resp_sampler):
            for _ in range(len(batch_sampler)):
                self.resp_perm.append(resp_id)
        self.shuffle = shuffle

    @abc.abstractmethod
    def get_inner_sampler(self, resp_dataset):
        pass

    def __len__(self):
        return len(self.resp_perm)

    def __iter__(self):
        # XXX: Not using generator here...
        if self.shuffle:
            self.random.shuffle(self.resp_perm)
        resp_iters = [iter(sampler) for sampler in self.resp_sampler]
        for resp_id in self.resp_perm:
            yield self.make_batch(resp_id, resp_iters[resp_id])

    def make_batch(self, resp_id, resp_iter):
        return list(zip(repeat(resp_id), next(resp_iter)))


class SVL12KLoaderBase(pl.LightningDataModule):
    def __init__(
        self,
        *,
        vocab_response_path: str,
        stoi,
        splitter=fixed_split_ehara,
        multiple_thresh: bool = False,
    ):
        super().__init__()
        self.vocab_response_path = vocab_response_path
        self.stoi = stoi
        self.splitter = splitter
        self.multiple_thresh = multiple_thresh

    def _mk_ds(self, df, multiple_thresh=False):
        return SVL12KDataset(self.stoi, df, multiple_thresh=multiple_thresh)

    def _mk_dl(self, ds, **kwargs):
        return DataLoader(
            ds,
            batch_sampler=self.make_batch_sampler(ds, **kwargs),
            collate_fn=self.collate,
            pin_memory=True,
            num_workers=min(
                cpu_count(only_physical_cores=True),
                int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
            ),
            persistent_workers=True,
        )

    def setup(self, stage=None):
        df = pandas.read_parquet(self.vocab_response_path)
        train_df, val_df, test_df = self.splitter(df)
        self.test_ds = self._mk_ds(test_df.reset_index())
        self.train_ds = self._mk_ds(train_df.reset_index())
        self.val_ds = self._mk_ds(val_df.reset_index())
        if self.multiple_thresh:
            self.multiple_thresh_train_ds = self._mk_ds(
                train_df.reset_index(), multiple_thresh=True
            )

    def train_dataloader(self):
        return self._mk_dl(self.train_ds)

    def val_dataloader(self):
        return self._mk_dl(self.val_ds)

    def test_dataloader(self):
        return self._mk_dl(self.test_ds)


NON_ALPHANUMERIC = re.compile(r"[\W_]+", re.UNICODE)


def apply_stoi(stoi, word):
    res = stoi.get(word)
    if res is not None:
        return res
    word = word.lower()
    res = stoi.get(word)
    if res is not None:
        return res
    word = NON_ALPHANUMERIC.sub("", word)
    res = stoi.get(word)
    if res is not None:
        return res
    word = word[:-1]
    # TODO: Last resort OOV strategy
    return stoi[word]
