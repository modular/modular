# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# We can run this file with various targets.
# RUN: not mojo -target-triple not-a-valid-target %s 2>&1 | FileCheck %s --check-prefix=INVALID_TARGET
# INVALID_TARGET: mojo: error: failed to create target info: unknown target triple 'not-a-valid-target'

# RUN: %mojo %s 2>&1 | FileCheck %s


def main():
    # CHECK: hello world
    print("hello world")
