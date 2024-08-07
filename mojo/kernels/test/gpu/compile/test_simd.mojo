# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s
from gpu.host._compile import _compile_code
from testing import assert_true


def test_add():
    fn add[
        width: Int
    ](x: SIMD[DType.bfloat16, width], y: __type_of(x)) -> __type_of(x):
        return x + y

    assert_true("fma.rn.bf16 " in _compile_code[add[width=1]]().asm)
    assert_true("fma.rn.bf16x2 " in _compile_code[add[width=2]]().asm)
    assert_true("fma.rn.bf16x2 " in _compile_code[add[width=8]]().asm)


def test_sub():
    fn sub[
        width: Int
    ](x: SIMD[DType.bfloat16, width], y: __type_of(x)) -> __type_of(x):
        return x - y

    assert_true("fma.rn.bf16 " in _compile_code[sub[width=1]]().asm)
    assert_true("fma.rn.bf16x2 " in _compile_code[sub[width=2]]().asm)
    assert_true("fma.rn.bf16x2 " in _compile_code[sub[width=8]]().asm)


def test_mul():
    fn mul[
        width: Int
    ](x: SIMD[DType.bfloat16, width], y: __type_of(x)) -> __type_of(x):
        return x * y

    assert_true("fma.rn.bf16 " in _compile_code[mul[width=1]]().asm)
    assert_true("fma.rn.bf16x2 " in _compile_code[mul[width=2]]().asm)
    assert_true("fma.rn.bf16x2 " in _compile_code[mul[width=8]]().asm)


def test_fma():
    fn fma[
        width: Int
    ](
        x: SIMD[DType.bfloat16, width], y: __type_of(x), z: __type_of(x)
    ) -> __type_of(x):
        return x * y + z

    assert_true("fma.rn.bf16 " in _compile_code[fma[width=1]]().asm)
    assert_true("fma.rn.bf16x2 " in _compile_code[fma[width=2]]().asm)
    assert_true("fma.rn.bf16x2 " in _compile_code[fma[width=8]]().asm)


def main():
    test_add()
    test_sub()
    test_mul()
    test_fma()
