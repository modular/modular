# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO: Enable once https://github.com/modularml/modular/issues/37376 is resolved
# UNSUPPORTED: asan
# RUN: %mojo-no-debug %s

from subprocess import run

from testing import *


def main():
    assert_not_equal(run("ls"), "")
