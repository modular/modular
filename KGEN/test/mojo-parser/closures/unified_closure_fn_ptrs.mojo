# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# COM: Verify generated wrapper structure

# CHECK: lit.trait.decl @"def(x: Int) -> Int"
# CHECK: lit.struct.decl @"def(x: Int) -> Int_PtrWrapper"<Impl: !lit.generator<("x": !Int1) -> !Int1>

# CHECK: lit.fn @"__call__
# CHECK: %1 = lit.call tail[!lit.generator<("x": !Int1) -> !Int1>: Impl](%x)
# CHECK: lit.return %1 : !Int1
# CHECK: lit.end_fn

# CHECK: kgen.conformance @"def(x: Int) -> Int"
# CHECK: kgen.witness "__call__($0,::Int)"

# CHECK: kgen.conformance @"{{.*}}::AnyType" {
# CHECK-NEXT: }

# CHECK: lit.fn @"wrap_fn()"
# CHECK: %__call_result_tmp__ = lit.var.decl "__call_result_tmp__" synth
# CHECK: %0 = lit.call {{.*}}:@"def(x: Int) -> Int_PtrWrapper"::@"__init__()"
# CHECK: %1 = lit.ref.immut %__call_result_tmp__


def top_level(x: Int) -> Int:
    return x


def use_closure[Impl: def(x: Int) unified -> Int](cb: Impl) -> Int:
    return cb(1)


def wrap_fn() -> Int:
    return use_closure(top_level)


# // -----

# COM: Verify that wrappers are deduplicated

# CHECK-COUNT-1: lit.struct.decl @"def(x: Int) -> Int_PtrWrapper"


def a(x: Int) -> Int:
    return x


def b(x: Int) -> Int:
    return x * x


def use_closure[Impl: def(x: Int) unified -> Int](cb: Impl) -> Int:
    return cb(1)


def wrap_fn() -> Int:
    return use_closure(a) + use_closure(b)


# // -----

# COM: fn literals can be converted to closure wrappers.

# CHECK: kgen.conformance @"def(x: Int) -> Int"
# CHECK: kgen.conformance @"def(Int) -> Int"


def top_level(x: Int) -> Int:
    return x


# COM: Note the lack of an argument name in the signature.
def use_closure[Impl: def(Int) unified -> Int](cb: Impl) -> Int:
    return cb(1)


def wrap_fn() -> Int:
    var _ = use_closure(top_level)
    var _ = use_closure[type_of(top_level)](top_level)
