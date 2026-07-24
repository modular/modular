# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s -mlir-print-op-generic | FileCheck %s

# COM: A lifted closure that captures a value whose type depends on a
# COM: compile-time (origin) parameter of the enclosing function embeds that
# COM: parameter in its closure-storage self type. That self type is part of
# COM: the closure's `funcTypeGenerator` signature, which must be
# COM: self-contained: references to the closure's own parameters must be
# COM: positional index refs (`*(0,0)`), not by name (`*"x`"`), so the
# COM: signature is valid when viewed in isolation. The generic op printer is
# COM: used because the pretty printer renders `functionType`, not
# COM: `funcTypeGenerator`.


struct Thing[a: Origin[mut=True]](TrivialRegisterPassable):
    pass


struct Foo(Movable where False):
    pass


def use(y: Thing):
    pass


def capture_implicit_origin(var x: Foo, y: Thing[origin_of(x)]):
    # The promoted `capture_it` closure captures `y : Thing[origin_of(x)]`, so
    # its closure-storage self type is parameterized by the hoisted origin `x`.
    # In the `funcTypeGenerator` that origin must be the positional index ref
    # `*(0,0)`, not the named ref `*"x`"`.
    #
    # CHECK: funcTypeGenerator = !kgen.generator<!lit.generator<<"x`": origin<true>, +>[1](!lit.ref<!lit.struct<{{#[A-Za-z0-9_]+}} <:origin<true> *(0,0)>>{{.*}}sym_name = "capture_it()`
    def capture_it() {read y}:
        use(y)
