# VocabIRT

This repository accompanies the paper:

Robertson, F. "Word discriminations for vocabulary inventory prediction." In *Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP)* 2021 (In press).

## Reproduce experiments

The aim of the repository is to allow for the reproduction of the experiments
in the paper. To this end there is a Docker image you can use (it can also run
under Singularity). You can also used the `Dockerfile` as a reference for
installing the requirements manually (`poetry install` should get you most of
the way there).

First you need to process the datasets with
[vocabaqdata](https://github.com/frankier/vocabaqdata). Before that, you will
need to manually obtain the raw data according to the instructions in that
project's README. Note that testyourvocab can only be obtained by direct
communication. The other experiments can still be completed without it.

Here's an example of how you might process the datasets with vocabaqdata (from the root of the vocabaqdata repo):

    $ EVKD1_RAW=/path/to/evkd1 SVL12K_RAW=/path/to/svl12k TESTYOURVOCAB_RAW=/path/to/testyourvocab snakemake -j1 all_evkd1 all_svl12k all_testyourvocab

Then you can run all preprocessing and experiments using Snakemake:

    $ snakemake -j1

For running individual parts, refer to `workflow/Snakefile`.

## CAT demo

You can also run the CAT demo and it will estimate your vocabulary level.

    $ poetry python -m vocabirt.activelearn
