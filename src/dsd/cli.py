from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich import print as rprint

from dsd import __version__
from dsd.logging import init_logger
from dsd.scrnaseq import clean_scrnaseq

app = typer.Typer(
    name="dsd_clean",
    help="cli for dsd",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


def version_callback(version: Annotated[bool, typer.Option("--version")] = False) -> None:  # FBT001
    if version:
        rprint(f"[yellow]{__package__}[/] version: [bold blue]{__version__}[/]")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
@app.command(no_args_is_help=True, name="dsd")
def cli(
    sample_path: Annotated[
        Path | None,
        typer.Option(
            "--sample_path",
            help="Folder in which a single pair of 'sample_filtered_feature_bc_matrix.h5' and 'raw_feature_bc_matrix.h5' can be found.",
        ),
    ] = None,
    sample_matrix: Annotated[
        Path | None,
        typer.Option("-f", "--filtered_matrix", help="The filtered sample matrix produced by Cell Ranger."),
    ] = None,
    raw_sample_matrix: Annotated[
        Path | None,
        typer.Option("-r", "--raw_matrix", help="The raw sample matrix produced by Cell Ranger."),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            help="Control output verbosity. Pass this argument multiple times to increase the amount of output.",
            count=True,
        ),
    ] = 0,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show version", callback=version_callback, is_eager=True),
    ] = False,
):
    init_logger(verbose)
    if sample_path:
        sample_matrix = next(sample_path.joinpath("outs", "per_sample_outs").iterdir()).joinpath(
            "count", "sample_filtered_feature_bc_matrix.h5"
        )
        raw_sample_matrix = sample_path.joinpath("outs/multi/count/raw_feature_bc_matrix.h5")

    match (sample_matrix, raw_sample_matrix):
        case (False, False):
            msg = "Neither the sample_matrix nor the raw_sample_matrix were found. If you used `sample_path`, try explictly specifying the paths to these files"
            logger.error(msg)
            raise FileNotFoundError(msg)
        case (True, False):
            msg = "The raw_sample_matrix was not found."
            logger.error(msg)
            raise FileNotFoundError(msg)
        case (False, True):
            msg = "The sample_matrix was not found."
            logger.error(msg)
            raise FileNotFoundError(msg)
        case _:
            clean_scrnaseq(sample_matrix, raw_sample_matrix)


if __name__ == "__main__":
    logger.enable("dsd")
    app()  # pragma: no cover
