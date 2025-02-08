# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: L4-GPU || H100-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from gpu.host import DeviceContext
from memory import bitcast
from builtin.simd import *


fn print_bits[type: DType](val: Scalar[type]):
    var u8 = bitcast[DType.uint8](val)
    var bits = String()

    @parameter
    for i in reversed(range(8)):
        bits.write((u8 >> i) & 1)

    print(type, "nan:", u8, bits)


fn test():
    # CHECK: float8_e5m2 nan: 127 01111111
    print_bits(Float8_e5m2(FloatLiteral.nan))
    # CHECK: float8_e4m3fn nan: 127 01111111
    print_bits(Float8_e4m3(FloatLiteral.nan))


def main():
    with DeviceContext() as ctx:
        var func_8e4m3 = ctx.compile_function[test]()
        ctx.enqueue_function(func_8e4m3, grid_dim=1, block_dim=1)
