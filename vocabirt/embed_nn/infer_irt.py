import pickle

import click
import pandas
import torch

from embed_nn.cv_discrim import Predictor, apply_stoi_list
from embed_nn.utils import get_numberbatch_vec


def load_state_dict(vectors, checkpoint):
    data = torch.load(checkpoint)
    model_obj = Predictor(vectors=vectors, feature_size=2)
    model_obj.load_state_dict(data["state_dict"], strict=False)
    return model_obj


@click.command()
@click.argument("resp_in", type=click.File("rb"), nargs=2)
@click.argument("model")
@click.argument("out", type=click.File("wb"))
def main(resp_in, model, out):
    words = set()
    for path in resp_in:
        words.update(pandas.read_parquet(path)["word"])
    words = list(words)
    vectors = get_numberbatch_vec()
    model_obj = load_state_dict(vectors, model)
    preds = model_obj(apply_stoi_list(vectors.stoi, words))
    preds = preds.detach().numpy()
    pickle.dump(
        {
            "words": pandas.DataFrame(
                {
                    "word": words,
                    "discrimination": preds[:, 1],
                    "difficulty": preds[:, 0],
                }
            )
        },
        out,
    )


if __name__ == "__main__":
    main()
