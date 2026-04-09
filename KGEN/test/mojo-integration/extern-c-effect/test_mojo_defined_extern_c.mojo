# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Tests that a Mojo function declared with the abi("C") effect is compiled
# with C ABI at its definition site.
#
# FloatPair ({float, float}, 8 bytes) is classified as SSE on x86-64 SysV and
# returned packed in XMM0.  The test verifies three scenarios:
#
#   1. Direct Mojo call: ConvertKGENCall applies C ABI coercion at the call
#      site; processCABIFunctionDefinition applies it at the definition entry
#      and exit.  Both sides must agree for the result to be correct.
#
#   2. Call through abi("C") pointer: taking &mojo_add_one yields the address
#      of the C-ABI function (modified in place, not a Mojo-ABI impl), so the
#      existing call_indirect coercion path also works correctly.
#
#   3. C invokes Mojo callback: the function pointer is passed to C land via
#      external_call; C calls through it using C ABI, which must match the
#      Mojo-defined function's compiled ABI.
#
# C reference: c_reference.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_effect_reference.lo %s -o %t.dir/test_mojo_defined_extern_c
# RUN: %t.dir/test_mojo_defined_extern_c | FileCheck %s

from ffi import external_call


@fieldwise_init
struct FloatPair(TrivialRegisterPassable):
    var x: Float32
    var y: Float32


def mojo_add_one(p: FloatPair) abi("C") -> FloatPair:
    return FloatPair(p.x + 1.0, p.y + 1.0)


def test_direct():
    # Direct Mojo-to-Mojo call through a abi("C") function definition.
    # ConvertKGENCall inserts C ABI coercion at the call site;
    # processCABIFunctionDefinition inserts it at the function entry/exit.
    var r = mojo_add_one(FloatPair(1.0, 2.0))
    print(r.x, r.y)


# CHECK: 2.0 3.0


def test_through_pointer():
    # Taking the address of a abi("C") function gives the C-ABI function's
    # address (in-place rewrite, no rename).  The existing call_indirect
    # coercion path then applies C ABI coercion at the call site.
    var fp: def(FloatPair) abi("C") -> FloatPair = mojo_add_one
    var r = fp(FloatPair(3.0, 4.0))
    print(r.x, r.y)


# CHECK: 4.0 5.0


def test_c_calls_mojo():
    # Pass the Mojo-defined abi("C") function to C as a callback.
    # C invokes it using C ABI; the function must have been compiled with C ABI
    # at its definition site for the result to be correct.
    var r = external_call["c_apply_float_pair_fn", FloatPair](
        mojo_add_one, FloatPair(5.0, 6.0)
    )
    print(r.x, r.y)


# CHECK: 6.0 7.0


def main():
    test_direct()
    test_through_pointer()
    test_c_calls_mojo()
