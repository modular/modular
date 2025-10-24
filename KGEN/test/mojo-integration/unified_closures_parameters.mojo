# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 8 string | FileCheck %s

from sys import argv


@fieldwise_init
struct Parameter[*, base: ImplicitlyCopyable & Writable](Copyable):
    var impl: base

    fn useIt(self):
        print(self.impl)


fn takeIt[f: fn () unified -> None](impl: f):
    impl()


fn captureIt(p: Parameter[**_]):
    @no_inline
    fn closure() unified {read p}:
        p.useIt()

    takeIt(closure)


def main():
    var num = atol(argv()[1])
    var str = argv()[2]
    var p1 = Parameter[base=String](str)
    var p2 = Parameter[base=Int](num)
    # CHECK: string
    captureIt(p1)
    # CHECK: 8
    captureIt(p2)
