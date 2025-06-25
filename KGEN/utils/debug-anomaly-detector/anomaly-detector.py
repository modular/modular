# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from itertools import chain
from pathlib import Path

import click
from modular.utils.debuglib.debugger import get_debugger, run_target
from modular.utils.debuglib.sources import (
    MojoBinary,
    MojoCompilationError,
    MojoSource,
)
from rich.console import Console

CONSOLE = Console()


def debug_main_fn(source: MojoSource) -> None:
    """Places a breakpoint at the `main` function, if available, and runs the
    program up to that point. Skips sources that fail to compile."""

    main_lines = list(source.find_lines_with_text("fn main("))
    if len(main_lines) == 0:
        return
    main_line = main_lines[0]

    CONSOLE.print("━" * CONSOLE.width)
    CONSOLE.print(f"🐞 Debugging {source.path}\n")
    try:
        with MojoBinary(source, suppress_build_output=True) as bin_file:
            target = get_debugger().CreateTarget(str(bin_file))
            assert target.IsValid()

            bp = target.BreakpointCreateByLocation(str(source.path), main_line)
            assert bp.GetNumLocations() == 1, (
                f"Couldn't set a breakpoint at {str(source.path)}:{main_line}"
            )

            ctx = run_target(target)

            ctx.handle_command("v")
    except MojoCompilationError:
        CONSOLE.print("Skipping due to compilation error\n")


@click.command()
@click.argument("input-dir-or-file", type=click.Path(exists=True))
def run(input_dir_or_file: Path) -> None:
    """Utility that programmatically tests basic debugging flows helping detect
    anomalies within a given input directory or file.

    Output lines that start with `error:` signal anomalies when debugging."""

    input = Path.resolve(Path(input_dir_or_file))

    if input.is_file():
        debug_main_fn(MojoSource(input))
    else:
        for file in chain(input.glob("**/*.mojo"), input.glob("**/*.🔥")):
            debug_main_fn(MojoSource(file))


if __name__ == "__main__":
    run()
