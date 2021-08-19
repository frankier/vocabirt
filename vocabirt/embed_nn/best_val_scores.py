import os
import re
import statistics
from os.path import join as pjoin

import click
import numpy
import torch


def iter_validation_infos(fold_dir):
    validation_dir = pjoin(fold_dir, "validation")
    for validation_tar in os.listdir(validation_dir):
        validation_tar_path = pjoin(validation_dir, validation_tar)
        validation_info = torch.load(validation_tar_path)
        yield validation_info


def metadata_from_name(fold_dir):
    match = re.match("resp([0-9]+)_vocab([0-9]+)_rep([0-9]+)", fold_dir)
    if match is None:
        return None
    return {
        "resp_split_idx": int(match[1]),
        "vocab_split_idx": int(match[2]),
        "repetition_idx": int(match[3]),
    }


def get_range(ind):
    maxes = {
        "resp_split_idx": -1,
        "vocab_split_idx": -1,
        "repetition_idx": -1,
    }
    for fold_dir in os.listdir(ind):
        for validation_info in iter_validation_infos(ind, fold_dir):
            if "metadata" in validation_info:
                return validation_info["metadata"]
        for k, v in metadata_from_name(fold_dir).items():
            if v > maxes[k]:
                maxes[k] = v
    return {
        "effective_resp_splits": maxes["resp_split_idx"] + 1,
        "effective_vocab_splits": maxes["vocab_split_idx"] + 1,
        "num_repetitions": maxes["repetition_idx"] + 1,
    }


def print_dist(dist):
    print("dist", dist)
    print("dist sorted", sorted(dist))
    print("min", min(dist))
    print("mean", statistics.mean(dist))
    print("med", statistics.median(dist))
    print("max", max(dist))
    print("range", max(dist) - min(dist))
    if len(dist) > 1:
        sd = statistics.stdev(dist)
    else:
        sd = 0
    print("stdev", sd)


def best_epoch(ind):
    metadata = None
    best_info = None
    for validation_info in iter_validation_infos(ind):
        if "metadata" in validation_info:
            metadata = validation_info["metadata"]
        if (
            best_info is None
            or validation_info["train_val_mean_auroc"]
            > best_info["train_val_mean_auroc"]
        ):
            best_info = validation_info
    return best_info, metadata


def iter_best_epochs(ind):
    for fold_dir in os.listdir(ind):
        metadata = metadata_from_name(fold_dir)
        if metadata is None:
            continue
        best_info, metadata = best_epoch(pjoin(ind, fold_dir))
        yield fold_dir, metadata, best_info


@click.command()
@click.argument("ind")
@click.argument("outf", required=False, type=click.File("wb"))
def main(ind, outf):
    split_range = get_range(ind)
    total_range = (
        split_range["effective_resp_splits"],
        split_range["effective_vocab_splits"],
        split_range["num_repetitions"],
    )
    auroc_mat = numpy.zeros(total_range)
    epoch_mat = numpy.zeros(total_range, dtype=numpy.int32)
    for fold_dir, metadata, best_info in iter_best_epochs(ind):
        idx = (
            metadata["resp_split_idx"],
            metadata["vocab_split_idx"],
            metadata["repetition_idx"],
        )
        auroc_mat[idx] = best_info["train_val_mean_auroc"]
        epoch_mat[idx] = best_info["epoch"]
    print("epoch_mat")
    print(epoch_mat)
    print("first split dist")
    print_dist(auroc_mat[0, 0, :])
    print("splits mean")
    print_dist(auroc_mat.mean(axis=-1).ravel())
    print("vocab splits")
    print_dist(auroc_mat.mean(axis=(0, -1)))
    print("resp splits")
    print_dist(auroc_mat.mean(axis=(1, -1)))
    best_reps = numpy.argmax(auroc_mat, axis=-1)
    print("best reps")
    print(best_reps)
    if outf:
        torch.save(torch.as_tensor(best_reps), outf)


if __name__ == "__main__":
    main()
