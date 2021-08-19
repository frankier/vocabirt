from os.path import dirname

import click
from torchtext.vocab import Vectors


@click.command()
@click.argument("inf", type=click.Path())
def main(inf):
    """
    Process vectors in word2vec format (inc. ConceptNet) into a format which
    can directly be loaded by torchtext.
    """
    Vectors(name=inf, cache=dirname(inf))


if __name__ == "__main__":
    main()
