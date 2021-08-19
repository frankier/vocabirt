import os
import pickle
from os.path import join as pjoin

import click

from vocabirt.embed_nn.loader.common import get_resp_split

from .discrimexp import Scorer, evaluate, prepare_args, print_summary, strategy_opt


@click.command()
@click.argument("vocab_response_path")
@click.argument("irt_path_cv", type=click.Path())
@click.argument("outf", type=click.File("wb"))
@strategy_opt
@click.option(
    "--estimator", type=click.Choice(["logistic", "hill-climb"]), default="hill-climb",
)
@click.option("--no-discrim-preds/--discrim-preds")
@click.option("--verbose/--terse")
def main(
    vocab_response_path,
    irt_path_cv,
    outf,
    strategy,
    estimator,
    no_discrim_preds,
    verbose,
):
    scorer = Scorer()
    irt_paths = {}
    for path in os.listdir(irt_path_cv):
        if not path.startswith("resp") or not path.endswith(".pkl"):
            continue
        irt_paths[int(path[4])] = pjoin(irt_path_cv, path)
    splits = len(irt_paths)
    assert splits in (1, 3)
    for split, irt_path in irt_paths.items():
        train_resp, test_resp = get_resp_split(15, split, splits)
        if splits == 1:
            test_resp = train_resp
        with open(irt_path, "rb") as irt_pkl:
            args = prepare_args(
                vocab_response_path,
                irt_pkl,
                strategy,
                estimator,
                no_discrim_preds=no_discrim_preds,
                no_reestimate=False,
                verbose=verbose,
                scorer=scorer,
            )
        for respondent in test_resp:
            evaluate(**args, respondent=respondent)
    df = scorer.as_df()
    last_df = scorer.as_df(last_only=True)
    print_summary(last_df)
    pickle.dump(
        {"df": df, "last_df": last_df,}, outf,
    )


if __name__ == "__main__":
    main()
