import math
import pickle

import click
import numpy
import pandas
import torch
from catsim.estimation import HillClimbingEstimator
from catsim.initialization import FixedPointInitializer
from catsim.irt import icc_hpc
from catsim.selection import LinearSelector, MaxInfoSelector, UrrySelector
from catsim.stopping import MaxItemStopper
from torchmetrics.functional import auroc, average_precision, matthews_corrcoef

from vocabirt.embed_nn.loader.cv import uniform_stratified_shuffle_split
from vocabmodel.utils.freq import add_freq_strata

from .activelearn import load_bank
from .estimators import LogisticEstimator, MinExpectedScaleSelector


class RespSim:
    def __init__(self, initializer, selector, estimator, stopper, verbose=False):
        self.initializer = initializer
        self.selector = selector
        self.estimator = estimator
        self.stopper = stopper
        self.verbose = verbose

    def run(self, words, bank, resp, logger=lambda *args: None):
        iter = 0
        theta = self.initializer.initialize()
        administered_items = []
        responses = []
        while 1:
            if self.verbose:
                print(f" == {iter + 1} == ")
                print("Current theta", theta)
            logger(iter, theta, administered_items, responses)
            item_index = self.selector.select(
                items=bank,
                administered_items=administered_items,
                est_theta=theta,
                response_vector=responses,
            )
            next_word = words[item_index]
            correct = resp[resp["word"] == next_word]["known"].iloc[0]
            if self.verbose:
                print(
                    f"{next_word}; diff={bank[item_index, 1]}; discr={bank[item_index, 0]}; corr={correct}"
                )
            administered_items.append(item_index)
            responses.append(correct)
            if self.verbose:
                print("responses", responses)
            theta = self.estimator.estimate(
                items=bank,
                administered_items=administered_items,
                response_vector=responses,
                est_theta=theta,
            )
            stop = self.stopper.stop(
                administered_items=bank[administered_items], theta=theta
            )
            iter += 1
            if stop:
                if self.verbose:
                    print("Done")
                break
        result = logger(iter, theta, administered_items, responses)
        if self.verbose:
            print("Final theta", theta)
        return theta, administered_items, responses, result


class Scorer:
    def __init__(self):
        self.cols = {}
        self.last_iter = 0

    def add_scores(self, respondent, iter, icc_bank, theta, known):
        result = {}

        def add_val(name, val):
            result[name] = val
            self.cols.setdefault(name, []).append(val)

        self.last_iter = max(self.last_iter, iter)
        add_val("respondent", respondent)
        add_val("iter", iter)
        add_val("theta", theta)
        preds = icc_hpc(theta, icc_bank)
        preds_tensor = torch.as_tensor(preds)
        targets_tensor = torch.as_tensor(known).bool()
        auroc_val = auroc(preds_tensor, targets_tensor, pos_label=1).item()
        add_val("auroc", auroc_val)
        pos_ap = average_precision(preds_tensor, targets_tensor, pos_label=True).item()
        add_val("pos_ap", pos_ap)
        neg_ap = average_precision(
            1 - preds_tensor, ~targets_tensor, pos_label=True
        ).item()
        add_val("neg_ap", neg_ap)
        mcc_val = matthews_corrcoef(preds_tensor, targets_tensor, num_classes=2).item()
        if math.isnan(mcc_val):
            mcc_val = 0
        preds_bool = (preds_tensor >= 0.5).int()
        extra_info = ""
        if preds_bool.all():
            extra_info = " (all 1)"
            mcc_val = 0
        if not preds_bool.any():
            extra_info = " (all 0)"
            mcc_val = 0
        add_val("mcc", mcc_val)
        return {**result, "extra_info": extra_info}

    def as_df(self, last_only=False):
        df = pandas.DataFrame(self.cols)
        if last_only:
            df = df[df["iter"] == self.last_iter].drop(columns="iter")
        return df


def evaluate(
    sim,
    df,
    scorer,
    words,
    bank,
    icc_bank,
    respondent,
    no_reestimate=False,
    bayesian=False,
):
    resp = df[df["respondent"] == respondent]
    known = resp["known"].to_numpy()

    def log(iter, theta, administered_items, responses):
        return scorer.add_scores(respondent, iter, icc_bank, theta, known)

    theta, administered_items, responses, final_result = sim.run(words, bank, resp, log)
    if bayesian:
        # XXX: Add a new iteration for restimate or what?
        if no_reestimate:
            final_theta = theta.mean
        else:
            estimator = HillClimbingEstimator(dodd=True)
            final_theta = estimator.estimate(
                items=bank,
                administered_items=administered_items,
                response_vector=responses,
                est_theta=theta.mean,
            )
    else:
        final_theta = theta
    print(
        "Respondent {respondent:2d} ({final_theta: .4f}): {auroc:.4f} {mcc:.4f}{extra_info}".format(
            **final_result, final_theta=final_theta
        )
    )


strategy_opt = click.option(
    "--strategy",
    type=click.Choice(["random", "max-info", "min-scale", "urry"]),
    default="random",
)


def create_sim(strategy, est_strat, strata, abilities, **kwargs):
    initializer = FixedPointInitializer(start=0)
    if est_strat == "logistic":
        estimator = LogisticEstimator(use_discriminations=False)
    else:
        estimator = HillClimbingEstimator(dodd=True)
    if strategy == "random":
        rng = numpy.random.default_rng(42)
        word_perm = uniform_stratified_shuffle_split(rng, (1, 40), strata.to_numpy(),)
        selector = LinearSelector(word_perm[0])
    elif strategy == "max-info":
        selector = MaxInfoSelector()
    elif strategy == "urry":
        selector = UrrySelector()
    else:
        assert strategy == "min-scale"
        selector = MinExpectedScaleSelector()
        estimator = LogisticEstimator()
    stopper = MaxItemStopper(max_itens=40)

    return RespSim(initializer, selector, estimator, stopper, **kwargs)


def load_vocab(vocab_response_path, thresh):
    df = pandas.read_parquet(vocab_response_path)
    add_freq_strata(df)
    strata = df[df["respondent"] == 0]["stratum"]
    df = df[df["respondent"] != 0]
    df["known"] = df["score"] >= thresh

    return df, strata


def prepare_args(
    vocab_response_path,
    irt_path,
    strategy,
    estimator,
    no_discrim_preds,
    verbose=False,
    thresh=5,
    **kwargs,
):
    words, bank, abilities = load_bank(irt_path)
    df, strata = load_vocab(vocab_response_path, thresh)
    sim = create_sim(strategy, estimator, strata, abilities, verbose=verbose)

    if no_discrim_preds:
        icc_bank = bank.copy()
        icc_bank[:, 0] = 1
    else:
        icc_bank = bank
    bayesian = strategy.startswith("bayesian-")

    return {
        "sim": sim,
        "df": df,
        "words": words,
        "bank": bank,
        "icc_bank": icc_bank,
        "bayesian": bayesian,
        **kwargs,
    }


def print_summary(last_df):
    auroc_total = last_df["auroc"].mean()
    mcc_total = last_df["mcc"].mean()
    pos_ap_total = last_df["pos_ap"].mean(skipna=True)
    neg_ap_total = last_df["neg_ap"].mean(skipna=True)
    print(
        f"Means\tAUROC: {auroc_total}\tMCC: {mcc_total}\tAP+: {pos_ap_total}\tAP-: {neg_ap_total}"
    )


@click.command()
@click.argument("vocab_response_path")
@click.argument("irt_path", type=click.File("rb"))
@click.argument("outf", type=click.File("wb"))
@click.option("--respondent", type=int)
@strategy_opt
@click.option(
    "--estimator", type=click.Choice(["logistic", "hill-climb"]), default="hill-climb",
)
@click.option("--no-discrim-preds/--incl-discrim-preds")
@click.option("--no-reestimate/--reestimate")
@click.option("--verbose/--terse")
def main(
    vocab_response_path,
    irt_path,
    outf,
    respondent,
    strategy,
    estimator,
    no_discrim_preds,
    no_reestimate,
    verbose,
):
    scorer = Scorer()
    args = prepare_args(
        vocab_response_path,
        irt_path,
        strategy,
        estimator,
        no_discrim_preds,
        no_reestimate=no_reestimate,
        verbose=verbose,
        scorer=scorer,
    )
    if respondent is not None:
        evaluate(**args, respondent=respondent)
    else:
        for respondent in range(1, 16):
            evaluate(**args, respondent=respondent)
        df = scorer.as_df()
        last_df = scorer.as_df(last_only=True)
        print_summary(last_df)
        pickle.dump(
            {"df": df, "last_df": last_df,}, outf,
        )


if __name__ == "__main__":
    main()
