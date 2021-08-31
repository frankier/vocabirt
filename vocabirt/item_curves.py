import pickle

import click
import numpy
from labellines import labelLines
from matplotlib import pyplot as plt
from scipy import stats as ss

NUM_SAMPLE_POINTS = 2048
X_POINTS = numpy.linspace(-3, 3, NUM_SAMPLE_POINTS)


@click.command()
@click.argument("irts", type=click.File("rb"))
@click.argument("imgout")
@click.option("--ability", type=int, multiple=True)
@click.option("--word", type=str, multiple=True, required=True)
def main(irts, imgout, ability, word):
    irts_dict = pickle.load(irts)
    words = irts_dict["words"][irts_dict["words"]["word"].isin(word)]
    words.sort_values(by="difficulty", inplace=True)
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 3)
    for _, word in words.iterrows():
        discrimination = word["discrimination"]
        difficulty = word["difficulty"]
        word = word["word"]
        ax.plot(
            X_POINTS,
            ss.logistic.cdf(X_POINTS * discrimination - difficulty),
            label=word,
        )
    labelLines(ax.get_lines(), zorder=2.5)
    if ability:
        abilities = [irts_dict["abilities"][a] for a in ability]
    else:
        abilities = irts_dict["abilities"]
    for a in abilities:
        ax.plot(a, 0.5, "kx")
    ax.set_xlabel("difficulty / discrimination")
    ax.set_ylabel("known")
    fig.tight_layout()
    plt.savefig(imgout)


if __name__ == "__main__":
    main()
