#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import argparse
import json
import os
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
    parser.add_argument("--no-user", dest="user", action="store_false")
    parser.set_defaults(user=True)

    args = parser.parse_args()

    kernel_dir = Path(__file__).parent / "kernel"
    kernel_install_dir = Path(
        KernelSpecManager().install_kernel_spec(
            str(kernel_dir), "mojo-jupyter-kernel", user=args.user
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
            "--modular-path",
            str(os.environ["MODULAR_PATH"]),
        ],
        "language": "mojo",
        "codemirror_mode": "mojo",
        "language_info": {
            "name": "mojo",
            "mimetype": "text/x-mojo",
            "file_extension": ".mojo",
            "codemirror_mode": {"name": "mojo"},
        },
        "resources": {
            "logo-64x64": str(kernel_install_dir / "logo-64x64.png"),
            "logo-svg": str(kernel_install_dir / "logo.svg"),
        },
    }
    kernel_json_path = kernel_install_dir / "kernel.json"
    kernel_json_path.write_text(json.dumps(kernel_json, indent=2))


if __name__ == "__main__":
    main()
