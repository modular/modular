# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from subprocess import run

from testing import *


def main():
    assert_not_equal(run("ls"), "")
