# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s
from gpu.host._compile import _compile_code
from testing import assert_true


def test_add[dtype: DType]():
    fn add[width: Int](x: SIMD[dtype, width], y: __type_of(x)) -> __type_of(x):
        return x + y

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code[add[width=1]]().asm)
        assert_true("fma.rn.bf16x2 " in _compile_code[add[width=2]]().asm)
        assert_true("fma.rn.bf16x2 " in _compile_code[add[width=8]]().asm)
    else:
        assert_true("fma.rn.f16 " in _compile_code[add[width=1]]().asm)
        assert_true("fma.rn.f16x2 " in _compile_code[add[width=2]]().asm)
        assert_true("fma.rn.f16x2 " in _compile_code[add[width=8]]().asm)


def test_sub[dtype: DType]():
    fn sub[width: Int](x: SIMD[dtype, width], y: __type_of(x)) -> __type_of(x):
        return x - y

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code[sub[width=1]]().asm)
        assert_true("fma.rn.bf16x2 " in _compile_code[sub[width=2]]().asm)
        assert_true("fma.rn.bf16x2 " in _compile_code[sub[width=8]]().asm)
    else:
        assert_true("fma.rn.f16 " in _compile_code[sub[width=1]]().asm)
        assert_true("fma.rn.f16x2 " in _compile_code[sub[width=2]]().asm)
        assert_true("fma.rn.f16x2 " in _compile_code[sub[width=8]]().asm)


def test_mul[dtype: DType]():
    fn mul[width: Int](x: SIMD[dtype, width], y: __type_of(x)) -> __type_of(x):
        return x * y

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code[mul[width=1]]().asm)
        assert_true("fma.rn.bf16x2 " in _compile_code[mul[width=2]]().asm)
        assert_true("fma.rn.bf16x2 " in _compile_code[mul[width=8]]().asm)
    else:
        assert_true("fma.rn.f16 " in _compile_code[mul[width=1]]().asm)
        assert_true("fma.rn.f16x2 " in _compile_code[mul[width=2]]().asm)
        assert_true("fma.rn.f16x2 " in _compile_code[mul[width=8]]().asm)


def test_fma[dtype: DType]():
    fn fma[
        width: Int
    ](x: SIMD[dtype, width], y: __type_of(x), z: __type_of(x)) -> __type_of(x):
        return x * y + z

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code[fma[width=1]]().asm)
        assert_true("fma.rn.bf16x2 " in _compile_code[fma[width=2]]().asm)
        assert_true("fma.rn.bf16x2 " in _compile_code[fma[width=8]]().asm)
    else:
        assert_true("fma.rn.f16 " in _compile_code[fma[width=1]]().asm)
        assert_true("fma.rn.f16x2 " in _compile_code[fma[width=2]]().asm)
        assert_true("fma.rn.f16x2 " in _compile_code[fma[width=8]]().asm)


def main():
    test_add[DType.bfloat16]()
    test_add[DType.float16]()

    test_sub[DType.bfloat16]()
    test_sub[DType.float16]()

    test_mul[DType.bfloat16]()
    test_mul[DType.float16]()

    test_fma[DType.bfloat16]()
    test_fma[DType.float16]()
