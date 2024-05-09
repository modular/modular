# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s | FileCheck %s

from max.driver import Device


def main():
    var d = Device()
    # CHECK: -1
    print(d.info())
