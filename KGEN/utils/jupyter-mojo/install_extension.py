#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import shutil
from pathlib import Path

from modular.utils import logging, subprocess


def main() -> None:
    extension_dir = Path(__file__).parent / "extension"
    modular_dir = Path(__file__).parent.parent.parent.parent

    # Copy over the license file so that we can build the extension.
    shutil.copy(modular_dir / "LICENSE.md", extension_dir)

    # Install the python package.
    try:
        subprocess.run_shell_command(
            ["python3", "-m", "pip", "install", "-e", str(extension_dir)],
        )
    except:
        logging.critical("Failed to install the mojo_jupyter package.")
        exit(1)

    # Build the type script extension.
    try:
        subprocess.run_shell_command(
            ["jlpm", "run", "build"],
            cwd=extension_dir,
        )
    except:
        logging.critical(
            "Failed to build the mojo_jupyter typescript extension."
        )
        exit(1)

    # Link to the extension directory.
    try:
        subprocess.run_shell_command(
            [
                "jupyter",
                "labextension",
                "develop",
                str(extension_dir),
                "--overwrite",
            ],
            cwd=extension_dir,
        )
    except:
        logging.critical("Failed to install the mojo_jupyter extension.")
        exit(1)


if __name__ == "__main__":
    main()
