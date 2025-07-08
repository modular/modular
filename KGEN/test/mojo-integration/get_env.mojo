# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D VAR=1 %s | FileCheck %s

from sys import env_get_bool


def main():
    # CHECK: True
    print(env_get_bool["VAR", 0]())
