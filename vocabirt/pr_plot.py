import pickle

import click
import holoviews as hv
from bokeh.plotting import show


@click.command()
@click.argument("inf", type=click.File("rb"))
def main(inf):
    hv.extension("bokeh")
    hv.extension("matplotlib")
    prs = pickle.load(inf)
    curves = []
    for idx, (precision, recall, thresholds) in enumerate(prs["unknown_prs"]):
        curve = hv.Curve(
            {"Recall": recall, "Precision": precision},
            "Recall",
            "Precision",
            label=f"Respondent {idx + 1}",
        )
        curve.opts(ylim=(0, 1.05))
        curve.opts(width=2000, height=900, backend="bokeh")
        curve.opts(linewidth=0.5, backend="matplotlib")
        curves.append(curve)
    overlay = hv.Overlay(curves)
    overlay.opts(legend_position="right")
    # overlay.opts(legend_offset=(100, 0), backend="bokeh")
    show(hv.render(overlay, backend="bokeh"))


if __name__ == "__main__":
    main()
