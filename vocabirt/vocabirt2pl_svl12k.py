import pickle
import tempfile
from string import Template

import arviz
import click
import numpy as np
import pandas
import torch
from cmdstanpy import CmdStanModel

from embed_nn.loader.common import apply_stoi
from embed_nn.pred.common import get_embedding
from embed_nn.utils import get_numberbatch_vec

STAN_TEMPLATE = Template(
    """
data {
  // # Common
  // Number of words in total
  int<lower=1> nwords;
  // Number of students
  int<lower=1> nstud;
  // Number of observations
  int<lower=1> nobs;
  // Student for observation n
  int<lower=1,upper=nstud> stud[nobs];
  // Question for observation n
  int<lower=1,upper=nwords> word[nobs];

  $DATA_EXTRA
}

parameters {
  // # Abilities
  // ability of svl12k student j - mean ability
  real ability[nstud];

  // # Discriminations
  real<lower = 0> discriminations[nwords];

  $PARAM_EXTRA
}

$SECTION_EXTRA

model {
  // This scales logit function to be v. close to probit
  real D = 1.702;
  ability ~ std_normal();
  difficulties ~ std_normal();
  discriminations ~ normal(1.2, 0.25);

  $MODEL_EXTRA
}
"""
)

DATA_ORDINAL = """
// Number of categories
int<lower=1> ncat;
// Rating for observation n
int<lower=1,upper=ncat> resp[nobs];
"""

DATA_LOGIT = """
// Rating for observation n
int<lower=0,upper=1> resp[nobs];
"""

DIFFICULTIES = """
// # Difficulties
// difficulty for 2->3 for word k
real difficulties[nwords];
"""

DIFFICULTY_OFFSETS = """
// offset of difficulty levels of different categories from difficulty
real<lower = 0> difficulty_12_offset;
real<lower = 0> difficulty_23_offset;
real<lower = 0> difficulty_34_offset;
"""

TRANSFORM_DIFFICULTIES = """
transformed parameters {
  // # Difficulties transformed for ordered_logistic
  ordered[ncat - 1] difficulties_full[nwords];
  for (n in 1:nwords) {
    real d = difficulties[n];
    difficulties_full[n] = [
      d - difficulty_12_offset - difficulty_23_offset - difficulty_34_offset,
      d - difficulty_23_offset - difficulty_34_offset,
      d - difficulty_34_offset,
      d
    ]';
  }
}
"""

# TODO: Figure out how to add in D * factor to all equations without breaking
# catsim

ORDINAL_MODEL = Template(
    """
// offset of difficulty levels of different categories from difficulty
difficulty_12_offset ~ std_normal();
difficulty_23_offset ~ std_normal();
difficulty_34_offset ~ std_normal();

for (i in 1:nobs) {
  resp[i] ~ ordered_logistic(
    (ability[stud[i]]$EXTRA_ABIL) * discriminations[word[i]],
    difficulties_full[word[i]] * discriminations[word[i]]
  );
}
"""
)

LOGIT_MODEL = Template(
    """
for (i in 1:nobs) {
  resp[i] ~ bernoulli_logit(
    (ability[stud[i]]$EXTRA_ABIL - difficulties[word[i]]) * discriminations[word[i]]
  );
}
"""
)


def mk_stan_code(given_difficulties=False, ordinal=False, add_mean_ability=False):
    data_extra = []
    param_extra = []
    section_extra = []
    model_extra = []

    if add_mean_ability:
        param_extra.append("real mean_ability;")
        param_extra.append("mean_ability ~ std_normal();")
        extra_abil = " + mean_ability"
    else:
        extra_abil = ""

    if given_difficulties:
        data_extra.append(DIFFICULTIES)
    else:
        param_extra.append(DIFFICULTIES)

    if ordinal:
        data_extra.append(DATA_ORDINAL)
        section_extra.append(TRANSFORM_DIFFICULTIES)
        param_extra.append(DIFFICULTY_OFFSETS)
        model_extra.append(ORDINAL_MODEL.substitute(EXTRA_ABIL=extra_abil))
    else:
        data_extra.append(DATA_LOGIT)
        model_extra.append(LOGIT_MODEL.substitute(EXTRA_ABIL=extra_abil))
    return STAN_TEMPLATE.substitute(
        DATA_EXTRA="\n".join(data_extra),
        PARAM_EXTRA="\n".join(param_extra),
        SECTION_EXTRA="\n".join(section_extra),
        MODEL_EXTRA="\n".join(model_extra),
    )


class SVL12KProcessor:
    def __init__(self, df):
        self.num_resp = df["respondent"].nunique()
        self.num_words = df["word"].nunique()
        self.df = df

    def prepare_data(self, difficulties=None, ordinal=False):
        stoi = {word: idx + 1 for idx, word in enumerate(self.df["word"].unique())}
        resp_renumber = {
            resp: idx + 1 for idx, resp in enumerate(self.df["respondent"].unique())
        }

        res = {
            "nwords": self.num_words,
            "nstud": self.num_resp,
            "nobs": self.num_resp * self.num_words,
            "stud": [resp_renumber[resp_id] for resp_id in self.df["respondent"]],
            "word": [stoi[word] for word in self.df["word"]],
        }

        if difficulties is not None:
            res["difficulties"] = difficulties

        if ordinal:
            res["ncat"] = 5
            res["resp"] = self.df["score"].to_numpy()
        else:
            res["resp"] = (self.df["score"].to_numpy() >= 5).astype(np.int32)

        return res

    def extract_from_opt(
        self,
        params_fit,
        verbose=True,
        take_difficulties=True,
        ordinal=False,
        add_mean_ability=False,
    ):
        cursor = 0
        arr = params_fit.optimized_params_np

        def take_slice(length):
            nonlocal cursor
            res = arr[cursor : cursor + length]
            cursor += length
            return res

        lp = take_slice(1)[0]
        ability = take_slice(self.num_resp)
        discriminations = take_slice(self.num_words)
        if add_mean_ability:
            mean_ability = take_slice(1)[0]
        if take_difficulties:
            difficulties = take_slice(self.num_words)
        if ordinal:
            difficulty_12_offset = take_slice(1)[0]
            difficulty_23_offset = take_slice(1)[0]
            difficulty_34_offset = take_slice(1)[0]

        if verbose:
            print("loss", lp)
            if add_mean_ability:
                print("mean_ability", mean_ability)
            print("abilities", ability)
            if ordinal:
                print("difficulty_12_offset", difficulty_12_offset)
                print("difficulty_23_offset", difficulty_23_offset)
                print("difficulty_34_offset", difficulty_34_offset)

        df_dict = {
            "word": self.df["word"].unique(),
            "discrimination": discriminations,
        }
        if take_difficulties:
            df_dict["difficulty"] = difficulties
        word_df = pandas.DataFrame(df_dict)

        res = {
            "loss": lp,
            "abilities": ability,
            "words": word_df,
        }
        if add_mean_ability:
            res["mean_ability"] = mean_ability
        if ordinal:
            res["difficulty_offsets"] = [
                -difficulty_12_offset - difficulty_23_offset - difficulty_34_offset,
                -difficulty_23_offset - difficulty_34_offset,
                -difficulty_34_offset,
                0,
            ]
        return res

    def extract_from_mcmc(self, params_mcmc):
        return arviz.from_cmdstanpy(
            posterior=params_mcmc,
            log_likelihood="log_lik",
            coords={
                "words": np.arange(self.num_words),
                "students": np.arange(self.num_resp),
            },
            dims={
                "difficulties": ["words"],
                "discriminations": ["words"],
                "ability": ["students"],
            },
        )


def get_word_difficulties(words, difficulties):
    stoi = get_numberbatch_vec().stoi
    words_tensor = torch.as_tensor([apply_stoi(stoi, word) for word in words])
    embedding = get_embedding(difficulties, stoi=stoi, words=words_tensor)
    return embedding(words_tensor)[:, 0].numpy()


def get_difficulties(df, difficulties):
    words = df["word"].unique()
    return get_word_difficulties(words, difficulties)


def estimate_irt(
    df, *, difficulties=None, optimize=True, ordinal=False, add_mean_ability=False
):
    proc = SVL12KProcessor(df)
    stan_code = mk_stan_code(
        given_difficulties=difficulties is not None,
        ordinal=ordinal,
        add_mean_ability=add_mean_ability,
    )
    print(stan_code)
    with tempfile.NamedTemporaryFile("w+", suffix=".stan") as fp:
        fp.write(stan_code)
        fp.flush()

        if difficulties is None:
            data = proc.prepare_data(ordinal=ordinal)
        else:
            difficulties_vec = get_difficulties(df, difficulties)
            data = proc.prepare_data(difficulties_vec, ordinal=ordinal)

        model = CmdStanModel("irt", fp.name)
        model.compile()
        if optimize:
            stan_fit = model.optimize(data=data)
            if difficulties is None:
                result = proc.extract_from_opt(
                    stan_fit, take_difficulties=True, add_mean_ability=add_mean_ability
                )
            else:
                result = proc.extract_from_opt(
                    stan_fit, take_difficulties=False, add_mean_ability=add_mean_ability
                )
                result["words"].insert(1, "difficulty", difficulties_vec)
            return result
        else:
            stan_fit = model.sample(
                data=data,
                chains=8,
                parallel_chains=8,
                threads_per_chain=2,
                show_progress=True,
            )
            return (proc.extract_from_mcmc(stan_fit), stan_fit)


@click.command()
@click.argument("inf")
@click.argument("outf", type=click.File("wb"))
@click.option("--optimize/--sample")
@click.option("--ordinal/--binomial")
@click.option("--difficulties")
@click.option("--add-mean-ability/--no-add-mean-ability")
def main(inf, outf, optimize, difficulties, ordinal, add_mean_ability):
    df = pandas.read_parquet(inf)
    df = df[df["respondent"] != 0]
    pickle.dump(
        estimate_irt(
            df,
            difficulties=difficulties,
            optimize=optimize,
            ordinal=ordinal,
            add_mean_ability=add_mean_ability,
        ),
        outf,
    )


if __name__ == "__main__":
    main()
