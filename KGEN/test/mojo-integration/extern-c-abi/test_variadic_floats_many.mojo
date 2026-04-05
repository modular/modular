# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# C reference: c_abi_variadic_floats.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_variadic_floats_many
# RUN: %t.dir/test_variadic_floats_many | FileCheck %s
# CHECK: 90.0

# Note: Variadic C functions require the variadicType attribute on
# pop.external_call to generate correct LLVM IR with isVarArg=true.


def main():
    # Test: Pass 12 float values to stress SSE register exhaustion
    # On x86_64: first 8 in xmm0-xmm7, remaining 4 on stack
    # C function: sum = sum_of(val + count) for each val, returns sum + count
    # sum = (1+2+...+12) = 78, plus count (12) = 90
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_many_floats".value,
        fnType=__mlir_attr[`(!pop.scalar<si64>) -> !pop.scalar<f64>`,],
        _type=Float64,
    ](
        Int(12),
        Float64(1.0),
        Float64(2.0),
        Float64(3.0),
        Float64(4.0),
        Float64(5.0),
        Float64(6.0),
        Float64(7.0),
        Float64(8.0),
        Float64(9.0),
        Float64(10.0),
        Float64(11.0),
        Float64(12.0),
    )
    print(result)
