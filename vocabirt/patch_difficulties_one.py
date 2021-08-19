import click

from .patch_difficulties import patch_difficulty


@click.command()
@click.argument("inf", type=click.File("rb"))
@click.argument("outf", type=click.File("wb"))
def main(inf, outf):
    patch_difficulty(inf, outf)


if __name__ == "__main__":
    main()
