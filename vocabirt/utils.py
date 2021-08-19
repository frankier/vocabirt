from statistics import mean, stdev

from vocabirt.embed_nn.features import get_word_freqs
from vocabirt.embed_nn.loader.common import NON_ALPHANUMERIC
from vocabirt.embed_nn.features import (
    get_aoa_embedding,
    get_frequency_embedding,
    get_norm_frequency_embedding,
    get_prevalence_embedding,
)


def get_norm_freq_map(words):
    buckets = []
    for word, bucket in get_word_freqs():
        if word not in words:
            continue
        buckets.append(bucket)
    mean_bucket = mean(buckets)
    std_bucket = stdev(buckets)
    norm_word_buckets = {}
    for word, bucket in get_word_freqs():
        if word not in words:
            continue
        norm_word_buckets[word] = (bucket - mean_bucket) / std_bucket
    return norm_word_buckets


def get_embedding(embed_path, stoi=None, words=None, vocab_response_path=None):
    if stoi is None:
        stoi = get_numberbatch_vec().stoi
    if embed_path == "wordfreq":
        return get_frequency_embedding(stoi)
    elif embed_path == "wordfreq_norm":
        return get_norm_frequency_embedding(
            stoi, words=words, vocab_response_path=vocab_response_path
        )
    elif embed_path == "prevalence":
        return get_prevalence_embedding(stoi)
    elif embed_path == "kuperman_aoa":
        return get_aoa_embedding(stoi)
    else:
        return nn.Embedding.from_pretrained(torch.load(embed_path))
