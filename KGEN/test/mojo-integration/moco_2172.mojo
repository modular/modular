# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s

from collections import Set

alias keys: List[Int] = [1, 2, 3, 7, 5]
alias num_set = Set(keys)


def main():
    # CHECK: 5
    alias l = len(num_set)
    print(l)
    # CHECK: True
    alias contains = num_set.__contains__(7)
    print(contains)
