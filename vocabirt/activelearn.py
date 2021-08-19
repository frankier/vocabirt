import pickle

import click
import numpy


def most_confident_items(words, bank, theta):
    from catsim.irt import icc_hpc

    iccs = icc_hpc(theta, bank)
    iccs_pairs = sorted(zip(-iccs, words))
    print("Confidently known")
    for icc_neg, word in iccs_pairs[:50]:
        icc = -icc_neg
        print(f"{word}: {icc}")
    print("Confidently unknown")
    iccs_pairs = sorted(zip(iccs, words))
    for icc, word in iccs_pairs[:50]:
        print(f"{word}: {icc}")


def load_bank(irt_path):
    from catsim.irt import normalize_item_bank

    irt_info = pickle.load(irt_path)
    df = irt_info["words"]
    df_words = df["word"]
    bank = numpy.hstack(
        [
            df["discrimination"].to_numpy()[:, numpy.newaxis],
            df["difficulty"].to_numpy()[:, numpy.newaxis],
        ]
    )
    return df_words, normalize_item_bank(bank), irt_info.get("abilities")


@click.command()
@click.argument("irt_path", type=click.File("rb"))
def main(irt_path):
    from catsim.estimation import HillClimbingEstimator
    from catsim.initialization import RandomInitializer
    from catsim.irt import inf_hpc
    from catsim.selection import RandomesqueSelector
    from catsim.stopping import MaxItemStopper

    words, bank, abilities = load_bank(irt_path)
    initializer = RandomInitializer()
    selector = RandomesqueSelector(bin_size=1)
    estimator = HillClimbingEstimator(dodd=True)
    stopper = MaxItemStopper(20)
    administered_items = []
    responses = []
    theta = initializer.initialize()
    iter = 1
    while 1:
        print(f" == {iter} == ")
        print("Current theta", theta)
        infs = inf_hpc(theta, bank)
        organized_items = [
            (x, words[x], infs[x])
            for x in (-infs).argsort()
            if x not in administered_items
        ]
        print(organized_items[:10])
        item_index = selector.select(
            items=bank, administered_items=administered_items, est_theta=theta
        )
        print(
            "Item index", item_index, bank[item_index],
        )
        next_word = words[item_index]
        correct = (
            input(f"Do you know {next_word}? [y/n] ").strip().lower().startswith("y")
        )
        administered_items.append(item_index)
        responses.append(correct)
        theta = estimator.estimate(
            items=bank,
            administered_items=administered_items,
            response_vector=responses,
            est_theta=theta,
        )
        stop = stopper.stop(administered_items=bank[administered_items], theta=theta)
        if stop:
            print("Done")
            break
        iter += 1
    print("Final theta", theta)
    most_confident_items(words, bank, theta)


if __name__ == "__main__":
    main()
