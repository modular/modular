# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s --check-prefix=NO_ARGS
# RUN: %mojo %s --arg1 --arg2=10 --arg3="arg3" | FileCheck %s --check-prefix=ARGS

from IO import print
from Range import range
from Sys import argv


def main() -> None:
    # NO_ARGS: exec-argv.mojo

    # ARGS: exec-argv.mojo
    # ARGS: --arg1
    # ARGS: --arg2=10
    # ARGS: --arg3=arg3

    for i in range(argv().__len__()):
        print(argv()[i])
