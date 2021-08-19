import pickle

import click


@click.command()
@click.argument("pred_irt", type=click.File("rb"))
@click.argument("gold_irt", type=click.File("rb"))
@click.argument("out_irt", type=click.File("wb"))
@click.option("--pred-mean/--keep-pred")
def main(pred_irt, gold_irt, out_irt, pred_mean):
    cols = ["discrimination", "difficulty"]
    pred_df = pickle.load(pred_irt)["words"]
    gold_df = pickle.load(gold_irt)["words"]
    replaced_count = 0
    mean_discrimination = gold_df["discrimination"].mean()
    for idx, word in enumerate(pred_df["word"]):
        mask = gold_df["word"] == word
        if not mask.any():
            if pred_mean:
                pred_df.loc[idx, "discrimination"] = mean_discrimination
            continue
        pred_df.loc[idx, cols] = gold_df.loc[mask, cols].iloc[0]
        replaced_count += 1
    print(f"Replaced {replaced_count}/{len(pred_df)} rows")
    pickle.dump({"words": pred_df}, out_irt)


if __name__ == "__main__":
    main()
