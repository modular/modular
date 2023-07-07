# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo-driver run %s | FileCheck %s

from IO import print


def main() -> None:
    # CHECK: ok
    print("ok")
