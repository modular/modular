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


def create_argparser() -> argparse.ArgumentParser:
    """Helper for CL option definition and parsing logic."""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="sub-command help", dest="command")
    common_parser = argparse.ArgumentParser(add_help=False)

    parser_install = subparsers.add_parser(
        "install",
        help="Install the mojo jupyter kernel",
        parents=[common_parser],
    )
    parser_install.add_argument(
        "--python",
        default=sys.executable,
        help="The python interpreter to use when launching the kernel.",
    )
    parser_install.add_argument(
        "--no-user",
        dest="user",
        action="store_false",
        help="Install kernel to system-wide location",
    )
    parser_install.add_argument(
        "--modular-home", type=str, help="Modular home path"
    )
    parser_install.set_defaults(user=True, modular_home="")

    subparsers.add_parser(
        "uninstall",
        help="Uninstall the mojo jupyter kernel",
        parents=[common_parser],
    )

    return parser


def install_kernel(python: str, user: bool, modular_home: str):
    """Install the kernel spec."""
    kernel_dir = Path(__file__).parent / "kernel"
    kernel_install_dir = Path(
        KernelSpecManager().install_kernel_spec(
            str(kernel_dir), "mojo-jupyter-kernel", user=user
        )
    )

    if modular_home == "":
        modular_home = os.environ.get("MODULAR_HOME")
        if not modular_home:
            modular_home = os.environ.get("MODULAR_DERIVED_PATH")
        if not modular_home:
            raise RuntimeError("unable to resolve MODULAR_HOME path")

    # Generate the kernel.json file.
    kernel_json = {
        "display_name": "Mojo",
        "argv": [
            python,
            str(kernel_install_dir / "mojokernel.py"),
            "-f",
            "{connection_file}",
            "--modular-home",
            str(modular_home),
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


def uninstall_kernel():
    """Uninstall the kernel spec."""
    KernelSpecManager().remove_kernel_spec("mojo-jupyter-kernel")


def main():
    parser = create_argparser()
    args = parser.parse_args()

    if args.command == "install":
        install_kernel(args.python, args.user, args.modular_home)
    elif args.command == "uninstall":
        uninstall_kernel()
    else:
        raise Exception(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
