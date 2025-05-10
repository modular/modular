# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import argparse
import tempfile
from pathlib import Path

try:
    # Don't require including IPython as a dependency
    from IPython.core.magic import register_cell_magic  # type: ignore
except ImportError:

    def register_cell_magic(fn):
        return fn


from ..entrypoints.mojo import subprocess_run_mojo
from .paths import MojoCompilationError


@register_cell_magic
def mojo(line, cell):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="run")
    parser.add_argument("-o", "--output")

    args = parser.parse_args(line.strip().split())

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        mojo_path = path / "cell.mojo"
        with open(mojo_path, "w") as f:
            f.write(cell)
        (path / "__init__.mojo").touch()

        input_path = path if args.command == "package" else mojo_path
        command = [
            args.command,
            str(input_path),
            *(
                ("-o", args.output)
                if args.output and args.command != "run"
                else ()
            ),
        ]

        result = subprocess_run_mojo(command, capture_output=True)

    if not result.returncode:
        print(result.stdout.decode())
    else:
        raise MojoCompilationError(
            input_path, command, result.stdout.decode(), result.stderr.decode()
        )
