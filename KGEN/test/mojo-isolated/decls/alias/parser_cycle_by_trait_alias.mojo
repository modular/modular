# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# This program should be parsed without triggering a parser cycle.


trait BarAble:
    alias bar: Foo


# CHECK-LABEL: lit.struct.decl @BarViaTrait
struct BarViaTrait(BarAble):
    fn __init__(out self):
        pass

    alias bar: Foo = 10


struct Bar:
    @implicit
    fn __init__[B: BarAble](out self, b: B):
        pass


struct Foo:
    @implicit
    fn __init__(out self, value: Int):
        pass

    # This will force the body resolution ConformanceOp in BarViaTrait,
    # leading to the construction call emission for `bar: Foo = 10`
    fn __init__(out self, value: Bar = BarViaTrait()):
        pass
