# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# COM: Verify generated wrapper structure

# CHECK: lit.trait.decl @"fn(x: Int) -> Int"
# CHECK: lit.struct.decl @"fn(x: Int) -> Int_PtrWrapper"<Impl: !lit.generator<("x": !Int1) -> !Int1>

# CHECK: lit.fn @"__call__
# CHECK: %1 = lit.call[!lit.generator<("x": !Int1) -> !Int1>: Impl](%x)
# CHECK: lit.return %1 : !Int1
# CHECK: lit.end_fn

# CHECK: kgen.conformance @"fn(x: Int) -> Int"
# CHECK: kgen.witness "__call__($0,::Int)"

# CHECK: lit.fn @"wrap_fn()"
# CHECK: %wrappedFnPtr = lit.var.decl "wrappedFnPtr" var
# CHECK: %0 = lit.call {{.*}}:@"fn(x: Int) -> Int_PtrWrapper"::@"__init__()"
# CHECK: %1 = lit.ref.immut %wrappedFnPtr

fn top_level(x: Int) -> Int:
    return x


fn use_closure[Impl: fn(x: Int) unified -> Int](cb: Impl) -> Int:
    return cb(1)


fn wrap_fn() -> Int:
    return use_closure[top_level](top_level)

# // -----

# COM: Verify that wrappers are deduplicated

# CHECK-COUNT-1: lit.struct.decl @"fn(x: Int) -> Int_PtrWrapper"

fn a(x: Int) -> Int:
    return x

fn b(x: Int) -> Int:
    return x * x


fn use_closure[Impl: fn(x: Int) unified -> Int](cb: Impl) -> Int:
    return cb(1)


fn wrap_fn() -> Int:
    return use_closure[a](a) + use_closure[b](b)

# // -----

# COM: Wrappers should be rebound if signatures are compatible.

# CHECK: kgen.conformance @"fn(x: Int) -> Int"
# CHECK: kgen.conformance @"fn(Int) -> Int"

fn top_level(x: Int) -> Int:
    return x


# COM: Note the lack of an argument name in the signature.
fn use_closure[Impl: fn(Int) unified -> Int](cb: Impl) -> Int:
    return cb(1)


fn wrap_fn() -> Int:
    return use_closure[top_level](top_level)
