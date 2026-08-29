# RUN: not %mojo %s 2>&1 | FileCheck %s
from std.utils import IndexList


def main():
    # CHECK: IndexList: expected 3 elements, received 2
    var j: IndexList[3] = [1, 2]
