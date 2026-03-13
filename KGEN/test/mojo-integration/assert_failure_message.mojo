# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Verify that a failing assert prints the error message and file location.
# RUN: not %mojo %s 2>&1 | FileCheck %s


def main():
    # CHECK: Assert Error: x must be positive
    assert False, "x must be positive"
