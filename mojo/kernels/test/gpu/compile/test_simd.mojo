# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from testing import assert_true
from sys import has_nvidia_gpu_accelerator
from sys.info import _accelerator_arch


def test_add[dtype: DType]():
    fn add[width: Int](x: SIMD[dtype, width], y: __type_of(x)) -> __type_of(x):
        return x + y

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code_asm[add[width=1]]())
        assert_true("fma.rn.bf16x2 " in _compile_code_asm[add[width=2]]())
        assert_true("fma.rn.bf16x2 " in _compile_code_asm[add[width=8]]())
    else:
        assert_true("fma.rn.f16 " in _compile_code_asm[add[width=1]]())
        assert_true("fma.rn.f16x2 " in _compile_code_asm[add[width=2]]())
        assert_true("fma.rn.f16x2 " in _compile_code_asm[add[width=8]]())


def test_sub[dtype: DType]():
    fn sub[width: Int](x: SIMD[dtype, width], y: __type_of(x)) -> __type_of(x):
        return x - y

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code_asm[sub[width=1]]())
        assert_true("fma.rn.bf16x2 " in _compile_code_asm[sub[width=2]]())
        assert_true("fma.rn.bf16x2 " in _compile_code_asm[sub[width=8]]())
    else:
        assert_true("fma.rn.f16 " in _compile_code_asm[sub[width=1]]())
        assert_true("fma.rn.f16x2 " in _compile_code_asm[sub[width=2]]())
        assert_true("fma.rn.f16x2 " in _compile_code_asm[sub[width=8]]())


def test_mul[dtype: DType]():
    fn mul[width: Int](x: SIMD[dtype, width], y: __type_of(x)) -> __type_of(x):
        return x * y

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code_asm[mul[width=1]]())
        assert_true("fma.rn.bf16x2 " in _compile_code_asm[mul[width=2]]())
        assert_true("fma.rn.bf16x2 " in _compile_code_asm[mul[width=8]]())
    else:
        assert_true("fma.rn.f16 " in _compile_code_asm[mul[width=1]]())
        assert_true("fma.rn.f16x2 " in _compile_code_asm[mul[width=2]]())
        assert_true("fma.rn.f16x2 " in _compile_code_asm[mul[width=8]]())


def test_mul_sm90[dtype: DType]():
    fn mul[width: Int](x: SIMD[dtype, width], y: __type_of(x)) -> __type_of(x):
        return x * y

    @parameter
    if dtype is DType.bfloat16:
        assert_true(
            "mul.rn.bf16 "
            in _compile_code_asm[
                mul[width=1], target = _get_gpu_target["sm_90"]()
            ]()
        )
        assert_true(
            "mul.rn.bf16x2 "
            in _compile_code_asm[
                mul[width=2], target = _get_gpu_target["sm_90"]()
            ]()
        )
        assert_true(
            "mul.rn.bf16x2 "
            in _compile_code_asm[
                mul[width=8], target = _get_gpu_target["sm_90"]()
            ]()
        )
    else:
        assert_true(
            "mul.rn.f16 "
            in _compile_code_asm[
                mul[width=1], target = _get_gpu_target["sm_90"]()
            ]()
        )
        assert_true(
            "mul.rn.f16x2 "
            in _compile_code_asm[
                mul[width=2], target = _get_gpu_target["sm_90"]()
            ]()
        )
        assert_true(
            "mul.rn.f16x2 "
            in _compile_code_asm[
                mul[width=8], target = _get_gpu_target["sm_90"]()
            ]()
        )


def test_fma[dtype: DType]():
    fn fma[
        width: Int
    ](x: SIMD[dtype, width], y: __type_of(x), z: __type_of(x)) -> __type_of(x):
        return x * y + z

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code_asm[fma[width=1]]())
        assert_true("fma.rn.bf16x2 " in _compile_code_asm[fma[width=2]]())
        assert_true("fma.rn.bf16x2 " in _compile_code_asm[fma[width=8]]())
    else:
        assert_true("fma.rn.f16 " in _compile_code_asm[fma[width=1]]())
        assert_true("fma.rn.f16x2 " in _compile_code_asm[fma[width=2]]())
        assert_true("fma.rn.f16x2 " in _compile_code_asm[fma[width=8]]())


def test_cast():
    fn cast[
        src_type: DType, dst_type: DType, width: Int
    ](src: SIMD[src_type, width]) -> SIMD[dst_type, width]:
        return src.cast[dst_type]()

    assert_true(
        "cvt.rn.f16x2.f32"
        in _compile_code_asm[
            cast[src_type = DType.float32, dst_type = DType.float16, width=4]
        ]()
    )
    assert_true(
        "cvt.rn.bf16x2.f32"
        in _compile_code_asm[
            cast[src_type = DType.float32, dst_type = DType.bfloat16, width=4]
        ]()
    )
    assert_true(
        "cvt.f32.bf16"
        in _compile_code_asm[
            cast[src_type = DType.bfloat16, dst_type = DType.float32, width=1]
        ]()
    )
    assert_true(
        "cvt.f32.bf16"
        in _compile_code_asm[
            cast[src_type = DType.bfloat16, dst_type = DType.float32, width=4]
        ]()
    )


def main():
    # FIXME(KERN-1436): Enable for SM_90 case.
    if has_nvidia_gpu_accelerator() and not (
        "nvidia:90" in _accelerator_arch()
    ):
        test_add[DType.bfloat16]()
        test_add[DType.float16]()

        test_sub[DType.bfloat16]()
        test_sub[DType.float16]()

        test_mul[DType.bfloat16]()
        test_mul[DType.float16]()

    test_mul_sm90[DType.bfloat16]()
    test_mul_sm90[DType.float16]()

    test_fma[DType.bfloat16]()
    test_fma[DType.float16]()

    test_cast()
