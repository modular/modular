#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    extension_dir = Path(__file__).parent / "extension"
    modular_dir = Path(__file__).parent.parent.parent.parent

    # Copy over the license file so that we can build the extension.
    shutil.copy(modular_dir / "LICENSE.md", extension_dir)

    # Install the python package.
    try:
        subprocess.check_call(
            ["python3", "-m", "pip", "install", "-e", str(extension_dir)],
        )
    except:
        logging.critical("Failed to install the mojo_jupyter package.")
        exit(1)

    # Build the type script extension.
    try:
        subprocess.check_call(
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
        subprocess.check_call(
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
