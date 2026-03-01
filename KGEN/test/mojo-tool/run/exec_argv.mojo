# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s --check-prefix=NO_ARGS
# RUN: %mojo %s --arg1 --arg2=10 --arg3="arg3" | FileCheck %s --check-prefix=ARGS

from std.sys import argv


def main() raises:
    # NO_ARGS: exec_argv.mojo

    # ARGS: exec_argv.mojo
    # ARGS: --arg1
    # ARGS: --arg2=10
    # ARGS: --arg3=arg3

    for i in range(argv().__len__()):
        print(argv()[i])
