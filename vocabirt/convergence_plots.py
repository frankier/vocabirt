import os
import pickle
from os.path import join as pjoin

import click
import matplotlib.pyplot as plt
import pandas
import seaborn as sns


def load_scores(scores, is_ext=False, load_last=False):
    dfs = []
    for path in os.listdir(scores):
        if not path.endswith(".pkl"):
            continue
        base = path[:-4]
        with open(pjoin(scores, path), "rb") as inf:
            res = pickle.load(inf)
            if load_last:
                df = res["last_df"]
            else:
                df = res["df"]
        if is_ext:
            if base.count(".") == 2:
                discrim_src, diff_src, discrim = base.split(".", 2)
                df["discrim_src"] = discrim_src
                df["diff_src"] = diff_src
                df["discrim"] = discrim == "discrim"
            else:
                discrim_src, discrim = base.split(".", 1)
                df["discrim_src"] = discrim_src
                df["diff_src"] = "irt"
                df["discrim"] = discrim == "discrim"
        else:
            split_mode, strategy, estimator, diff, discrim = base.split(".", 4)
            df["split_mode"] = split_mode
            df["strategy"] = strategy
            df["estimator"] = estimator
            df["diff"] = diff
            df["discrim"] = discrim == "discrim"
        dfs.append(df)
    return pandas.concat(dfs)


@click.command()
@click.argument("scores")
@click.argument("abilities", type=click.File("rb"))
@click.argument("outf")
def main(scores, abilities, outf):
    df = load_scores(scores)
    plt.rcParams.update(
        {
            "font.size": 5,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        },
    )
    g = sns.relplot(
        data=df,
        x="iter",
        y="theta",
        col="respondent",
        col_wrap=3,
        hue="split_mode",
        style="strategy",
        kind="line",
    )
    """
    h, l = g.axes.get_legend_handles_labels()
    g.axes.legend_.remove()
    g.fig.legend(h, l, ncol=2, bbox_to_anchor=[0.5, 0], loc=8)
    """
    handles, labels = g.axes[0].get_legend_handles_labels()
    g._legend.remove()
    g.fig.legend(
        handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0),
    )
    abilities = pickle.load(abilities)["abilities"]
    for ax, ability in zip(g.axes, abilities):
        ax.axhline(ability, ls="--")
    plt.tight_layout()
    fig = plt.gcf()
    fig.tight_layout()
    fig.set_size_inches(3, 5)
    fig.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.2, top=0.95, left=0.12)
    plt.savefig(outf, dpi=600)


if __name__ == "__main__":
    main()
