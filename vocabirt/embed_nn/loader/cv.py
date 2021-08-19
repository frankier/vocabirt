from itertools import product

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .common import SVL12KLoaderBase, apply_stoi


def uniform_stratified_shuffle_split(rng, sizes, strata):
    main_sizes = sizes[:-1]
    sample_size = sizes[-1]
    strata_unique = np.unique(strata)
    num_strata_unique = len(strata_unique)
    out = np.zeros(sizes, dtype=np.long)
    for idx, stratum in enumerate(strata_unique):
        pool = np.nonzero(strata == stratum)[0]
        per_stratum_sample = sample_size // num_strata_unique
        """
        ValueError: Cannot take a larger sample than population when replace is False
        Can we go faster?
        out[
            tuple((slice(None) for _ in main_sizes))
            + (slice(idx * per_stratum_sample, (idx + 1) * per_stratum_sample),)
        ] = rng.choice(
            pool, (*main_sizes, per_stratum_sample), replace=False, shuffle=False
        )
        """
        for main_size in product(*(range(n) for n in main_sizes)):
            out[
                main_size
                + (slice(idx * per_stratum_sample, (idx + 1) * per_stratum_sample),)
            ] = rng.choice(pool, per_stratum_sample, replace=False, shuffle=False)
    return out


class BatchingDatasetWrapper(Dataset):
    def __init__(
        self,
        inner,
        stoi,
        support_size: int = 40,
        chunk_size: int = 1,
        steps_per_epoch: int = 1000,
        # XXX: Makes no difference. Is shuffled but always the same for every epoch.
        shuffle=True,
    ):
        self.inner = inner
        self.stoi = stoi
        self.support_size = support_size
        self.steps_per_epoch = steps_per_epoch
        num_respondents = len(self.inner.resp_datasets)
        ds = self.inner.resp_datasets[0]
        self.resp_perm = (
            list(range(num_respondents)) * (steps_per_epoch // num_respondents + 1)
        )[:steps_per_epoch]
        # XXX: Bad: only one permutation, reused across multiple workers
        # XXX: Should switch to iterable dataset
        rng = np.random.default_rng(42)
        rng.shuffle(self.resp_perm)

        # num_words = len(ds)
        # self.word_perm = rng.integers(
        # 0, num_words, (steps_per_epoch, chunk_size, support_size)
        # )
        self.word_perm = uniform_stratified_shuffle_split(
            rng,
            (steps_per_epoch, chunk_size, support_size),
            ds.df["stratum"].to_numpy(),
        )

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx: int):
        resp_idx = self.resp_perm[idx]
        support_idxs = self.word_perm[idx]
        df = self.inner.resp_datasets[resp_idx].df
        return (
            resp_idx,
            np.array([apply_stoi(self.stoi, word) for word in df["word"]]),
            (df["score"] >= 5).to_numpy(),
            support_idxs,
        )


def mk_cv_dl(ds, stoi, shuffle=True, support_size=30, chunk_size=4):
    return DataLoader(
        BatchingDatasetWrapper(
            ds, stoi, support_size=support_size, chunk_size=chunk_size, shuffle=shuffle,
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        # XXX: Don't increase this until shuffling is fixed
        num_workers=0,
        # persistent_workers=True,  # broken with pin_memory until 1.8.0
    )


class SVL12KLoader(SVL12KLoaderBase):
    def __init__(
        self,
        *,
        vocab_response_path: str,
        stoi,
        support_size: int = 40,
        chunk_size: int = 4,
        multiple_thresh: bool = False,
    ):
        super(SVL12KLoader, self).__init__(
            vocab_response_path=vocab_response_path,
            stoi=stoi,
            multiple_thresh=multiple_thresh,
        )
        self.stoi = stoi
        self.support_size = support_size
        self.chunk_size = chunk_size

    def _mk_dl(self, ds, shuffle=True):
        return mk_cv_dl(
            ds,
            self.stoi,
            shuffle=shuffle,
            support_size=self.support_size,
            chunk_size=self.chunk_size,
        )

    def val_dataloader(self):
        return self._mk_dl(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._mk_dl(self.test_ds, shuffle=False)
