#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Exits successfully if the file at the given path is empty or does"
            " not exist. Otherwise, prints the file's contents, then exits"
            " unsuccessfully."
        )
    )
    parser.add_argument("path")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        return

    with open(args.path, "r") as f:
        content = f.read().strip()
        if content:
            print(
                f"error: '{args.path}' is not empty:\n{content}",
                file=sys.stderr,
            )
            exit(1)


if __name__ == "__main__":
    main()
