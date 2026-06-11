# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

# Tests that an in-place `String` append (`+=`) whose result stays within the
# inline (SSO) buffer produces the same value at compile time (`comptime`) and
# at runtime.


def append_small() -> String:
    var s: String = "ab"  # stays inline (SSO)
    s += "cd"  # still inline
    return s


def main():
    # Runtime.
    var rt = append_small()
    # CHECK: abcd
    print(rt)

    # Compile time.
    comptime ct = append_small()
    # CHECK: abcd
    print(ct)
