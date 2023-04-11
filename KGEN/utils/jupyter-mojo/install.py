#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import argparse
import json
import sys
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="The python interpreter to use when launching the kernel.",
    )
    args = parser.parse_args()

    kernel_dir = Path(__file__).parent / "kernel"
    kernel_install_dir = Path(
        KernelSpecManager().install_kernel_spec(
            str(kernel_dir), "mojo-jupyter-kernel", user=True
        )
    )

    # Generate the kernel.json file.
    kernel_json = {
        "display_name": "Mojo",
        "argv": [
            args.python,
            str(kernel_install_dir / "mojokernel.py"),
            "-f",
            "{connection_file}",
        ],
        "language": "mojo",
        "language_info": {
            "name": "mojo",
            "mimetype": "text/x-mojo",
            "file_extension": ".mojo",
        },
    }
    kernel_json_path = kernel_install_dir / "kernel.json"
    kernel_json_path.write_text(json.dumps(kernel_json, indent=2))


if __name__ == "__main__":
    main()
