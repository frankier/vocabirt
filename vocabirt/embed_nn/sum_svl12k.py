import click
import pandas

from vocabmodel.utils.freq import add_freq_strata, print_df


@click.command()
@click.argument("vocab_response_path")
def main(vocab_response_path):
    df = pandas.read_parquet(vocab_response_path)
    add_freq_strata(df)
    df["knows"] = df["score"] == 5
    print_df(df.groupby(["respondent"]).agg(["mean", "count"]))
    print_df(df.groupby(["respondent", "stratum"]).agg(["mean", "count"]))
    print_df(df[["respondent", "knows"]].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
