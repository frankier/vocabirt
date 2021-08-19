import pickle

import click
import holoviews as hv
import hvplot
import hvplot.pandas  # noqa
import pandas as pd
from bokeh.io import export_svgs

from .results import load_scores_errs


def export_svg(obj, filename):
    plot_state = hv.renderer("bokeh").get_plot(obj).state
    plot_state.output_backend = "svg"
    export_svgs(plot_state, filename=filename)


pd.options.plotting.backend = "holoviews"


@click.command()
@click.argument("scores")
@click.argument("abilities", type=click.File("rb"))
@click.option("--out", type=click.Path(), required=False, multiple=True)
def main(scores, abilities, out):
    abilities = pickle.load(abilities)["abilities"]
    df = load_scores_errs(scores, abilities)
    df.reset_index(inplace=True)
    df_both = df[df["split_mode"] == "both"]
    plot_auroc = df_both.hvplot.violin(
        y="auroc", by=["strategy", "discrim", "diff"], ylabel="AUROC", rot=90,
    )
    plot_mcc = df_both.hvplot.violin(
        y="mcc", by=["strategy", "discrim", "diff"], ylabel="MCC", rot=90,
    )
    plot_abs_theta_err = df.hvplot.violin(
        y="abs_theta_err",
        by=["split_mode", "strategy"],
        ylabel="abs theta err",
        rot=90,
    )
    if out:
        export_svg(plot_auroc, out[0])
        export_svg(plot_mcc, out[1])
        export_svg(plot_abs_theta_err, out[2])
        # hv.save(plot_auroc, out[0])
        # hv.save(plot_abs_theta_err, out[1])
    else:
        hvplot.show(plot_auroc + plot_abs_theta_err)


if __name__ == "__main__":
    main()
