# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s

from collections import Set

comptime keys: List[Int] = [1, 2, 3, 7, 5]
comptime num_set = Set(keys)


def main():
    # CHECK: 5
    comptime l = len(num_set)
    print(l)
    # CHECK: True
    comptime contains = num_set.__contains__(7)
    print(contains)
