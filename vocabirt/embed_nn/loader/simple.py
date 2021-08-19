import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from .common import RespondentPermSamplerBase, SVL12KLoaderBase, fixed_split_ehara


class RespondentBatchSampler(RespondentPermSamplerBase):
    def __init__(self, dataset, batch_size, generator=None, shuffle=True):
        self.batch_size = batch_size
        super(RespondentBatchSampler, self).__init__(
            dataset=dataset, generator=generator, shuffle=shuffle
        )

    def get_inner_sampler(self, resp_dataset):
        if self.shuffle:
            sampler = RandomSampler(resp_dataset, generator=self.generator)
        else:
            sampler = SequentialSampler(resp_dataset)
        return BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)


class SVL12KLoader(SVL12KLoaderBase):
    def __init__(
        self,
        *,
        vocab_response_path: str,
        stoi,
        splitter=fixed_split_ehara,
        batch_size: int = 40,
        multiple_thresh: bool = False,
    ):
        self.batch_size = batch_size
        super(SVL12KLoader, self).__init__(
            vocab_response_path=vocab_response_path,
            stoi=stoi,
            splitter=splitter,
            multiple_thresh=multiple_thresh,
        )

    def make_batch_sampler(self, ds):
        return RespondentBatchSampler(ds, self.batch_size)

    def collate(self, batch):
        # Not this absolutely assumes a single respondent per-batch
        # XXX: Perf can probably be improved easily here
        resp_idx = batch[0][3]
        return (
            resp_idx,
            torch.tensor([word for word, _, _, _ in batch]),
            torch.tensor([knows for _, knows, _, _ in batch]),
        )

    def val_dataloader(self):
        from .cv import mk_cv_dl

        return mk_cv_dl(self.val_ds, self.stoi, shuffle=False, chunk_size=1)

    def test_dataloader(self):
        from .cv import mk_cv_dl

        return mk_cv_dl(self.test_ds, self.stoi, shuffle=False, chunk_size=1)
