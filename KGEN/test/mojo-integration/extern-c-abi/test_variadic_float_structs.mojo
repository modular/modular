# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# C reference: c_abi_variadic_floats.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_variadic_float_structs
# RUN: %t.dir/test_variadic_float_structs | FileCheck %s

# Note: Variadic C functions require the variadicType attribute on
# pop.external_call to generate correct LLVM IR with isVarArg=true.


# Test 1: 4-byte float struct
@fieldwise_init
struct FloatStruct4(TrivialRegisterPassable):
    var a: Float32


# CHECK: variadic_float_4byte: 11.5
def test_variadic_float_4byte():
    var s = FloatStruct4(10.5)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_float_4byte".value,
        fnType=__mlir_attr[`(!pop.scalar<si64>) -> !pop.scalar<si64>`,],
        _type=FloatStruct4,
    ](Int(999), s)
    print("variadic_float_4byte:", result.a)


# Test 2: 8-byte float struct (two floats)
@fieldwise_init
struct FloatStruct8(TrivialRegisterPassable):
    var a: Float32
    var b: Float32


# CHECK: variadic_float_8byte: 11.5 21.5
def test_variadic_float_8byte():
    var s = FloatStruct8(10.5, 20.5)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_float_8byte".value,
        fnType=__mlir_attr[`(!pop.scalar<si64>) -> !pop.scalar<si64>`,],
        _type=FloatStruct8,
    ](Int(999), s)
    print("variadic_float_8byte:", result.a, result.b)


# Test 3: 8-byte double struct
@fieldwise_init
struct DoubleStruct8(TrivialRegisterPassable):
    var a: Float64


# CHECK: variadic_double_8byte: 101.5
def test_variadic_double_8byte():
    var s = DoubleStruct8(100.5)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_double_8byte".value,
        fnType=__mlir_attr[`(!pop.scalar<si64>) -> !pop.scalar<si64>`,],
        _type=DoubleStruct8,
    ](Int(999), s)
    print("variadic_double_8byte:", result.a)


# Test 4: 16-byte float struct (four floats)
@fieldwise_init
struct FloatStruct16(TrivialRegisterPassable):
    var a: Float32
    var b: Float32
    var c: Float32
    var d: Float32


# CHECK: variadic_float_16byte: 11.5 21.5 31.5 41.5
def test_variadic_float_16byte():
    var s = FloatStruct16(10.5, 20.5, 30.5, 40.5)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_float_16byte".value,
        fnType=__mlir_attr[`(!pop.scalar<si64>) -> !pop.scalar<si64>`,],
        _type=FloatStruct16,
    ](Int(999), s)
    print("variadic_float_16byte:", result.a, result.b, result.c, result.d)


# Test 5: 16-byte double struct (two doubles)
@fieldwise_init
struct DoubleStruct16(TrivialRegisterPassable):
    var a: Float64
    var b: Float64


# CHECK: variadic_double_16byte: 101.5 201.5
def test_variadic_double_16byte():
    var s = DoubleStruct16(100.5, 200.5)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_double_16byte".value,
        fnType=__mlir_attr[`(!pop.scalar<si64>) -> !pop.scalar<si64>`,],
        _type=DoubleStruct16,
    ](Int(999), s)
    print("variadic_double_16byte:", result.a, result.b)


# Test 6: 17-byte float struct (large, passed by pointer)
@fieldwise_init
struct FloatStruct17(TrivialRegisterPassable):
    var a: Float32
    var b: Float32
    var c: Float32
    var d: Float32
    var e: UInt8


# CHECK: variadic_float_17byte: 11.5 21.5 31.5 41.5 6
def test_variadic_float_17byte():
    var s = FloatStruct17(10.5, 20.5, 30.5, 40.5, 5)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_float_17byte".value,
        fnType=__mlir_attr[`(!pop.scalar<si64>) -> !pop.scalar<si64>`,],
        _type=FloatStruct17,
    ](Int(999), s)
    print(
        "variadic_float_17byte:",
        result.a,
        result.b,
        result.c,
        result.d,
        Int(result.e),
    )


def main():
    test_variadic_float_4byte()
    test_variadic_float_8byte()
    test_variadic_double_8byte()
    test_variadic_float_16byte()
    test_variadic_double_16byte()
    test_variadic_float_17byte()
