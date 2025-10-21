# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


from sys.info import _accelerator_arch, _is_sm_100x_or_newer

from gpu.host import get_gpu_target
from gpu.host.compile import _compile_code
from gpu.host.info import B200, GPUInfo
from testing import assert_false, assert_true


def test_operation[
    dtype: DType,
    target_arch: StaticString,
    op_fn: fn[width: Int] (x: SIMD[dtype, width], y: type_of(x)) -> type_of(x),
    op_name: StaticString,
]():
    var scalar: String
    var pairwise: String
    var suffix: String
    var scalar_ftz: String = ""

    # sm_80 does not support trivial add/sub/mul bfloat16 operations, but
    # these can be implemented using the FMA instruction. Verify that the
    # backend is using FMA and not falling back to widening the inputs to
    # float32.
    # sm_90 and later has wider support for bfloat16 operations.
    # sm_100 has support for f32x2 add/sub/mul/fma.
    var prefix: String

    @parameter
    if target_arch == "sm_80" and dtype is DType.bfloat16:
        prefix = "fma.rn"
    else:
        prefix = String(op_name)

    @parameter
    if dtype is DType.float16:
        suffix = ".f16"
    elif dtype is DType.float32:
        suffix = ".f32"
    else:
        suffix = ".bf16"

    # FTZ variants (will be accepted only when gated below).
    # Gate FTZ acceptance: only on Hopper+ *and* only for f16/f32.

    # On sm_80 or for bf16 (any arch), FTZ must NOT appear.
    if target_arch in ["sm_90", "sm_90a", "sm_100", "sm_100a", "sm_100b"] and (
        dtype is DType.float16 or dtype is DType.float32
    ):
        scalar_ftz = ".ftz"
    scalar = prefix + scalar_ftz + suffix
    pairwise = scalar + "x2 "

    alias target = get_gpu_target[target_arch]()
    var s1 = _compile_code[op_fn[width=1], target=target]()
    var s2 = _compile_code[op_fn[width=2], target=target]()
    var s8 = _compile_code[op_fn[width=8], target=target]()

    assert_true(scalar in s1)
    assert_true(pairwise in s2)
    assert_true(pairwise in s8)


def test_add[dtype: DType, target_arch: StaticString]():
    fn add[width: Int](x: SIMD[dtype, width], y: type_of(x)) -> type_of(x):
        return x + y

    test_operation[dtype, target_arch, add, "add"]()


def test_sub[dtype: DType, target_arch: StaticString]():
    fn sub[width: Int](x: SIMD[dtype, width], y: type_of(x)) -> type_of(x):
        return x - y

    test_operation[dtype, target_arch, sub, "sub"]()


def test_mul[dtype: DType, target_arch: StaticString]():
    fn mul[width: Int](x: SIMD[dtype, width], y: type_of(x)) -> type_of(x):
        return x * y

    test_operation[dtype, target_arch, mul, "mul"]()


def test_half_float_instruction_selection():
    def test_operations[dtype: DType, target_arch: StaticString]():
        test_add[dtype, target_arch]()
        test_sub[dtype, target_arch]()
        test_mul[dtype, target_arch]()

    def test_types[dtype: DType]():
        test_operations[dtype, "sm_80"]()
        test_operations[dtype, "sm_90"]()

    test_types[DType.bfloat16]()
    test_types[DType.float16]()


def test_fma[dtype: DType]():
    fn fma[
        width: Int
    ](x: SIMD[dtype, width], y: type_of(x), z: type_of(x)) -> type_of(x):
        return x * y + z

    fn fma_manual[
        width: Int
    ](x: SIMD[dtype, width], y: type_of(x), z: type_of(x)) -> type_of(x):
        return x.fma(y, z)

    # Ensure `_mul_with_fastmath_none` threads fastmath flags through to
    # pop.mul.
    fn _mul_with_fastmath_none[
        width: Int
    ](x: SIMD[DType.float32, width], y: type_of(x)) -> type_of(x):
        return x._mul_with_fastmath_none(y)

    @parameter
    if dtype is DType.bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code[fma[width=1]]())
        assert_true("fma.rn.bf16x2 " in _compile_code[fma[width=2]]())
        assert_true("fma.rn.bf16x2 " in _compile_code[fma[width=8]]())

        # Check that `_mul_with_fastmath_none` disables FMA contraction.
        assert_false(
            "fma.rn.bf16 " in _compile_code[_mul_with_fastmath_none[width=1]]()
        )
    elif dtype is DType.float32:
        var s1_f32 = _compile_code[fma_manual[width=1]]()
        var s2_f32 = _compile_code[fma_manual[width=2]]()
        var s8_f32 = _compile_code[fma_manual[width=8]]()
        # Allow FTZ on Hopper+;
        assert_true(("fma.rn.f32" in s1_f32) or ("fma.rn.ftz.f32" in s1_f32))
        assert_true(
            ("fma.rn.f32x2" in s2_f32) or ("fma.rn.ftz.f32x2" in s2_f32)
        )
        assert_true(
            ("fma.rn.f32x2" in s8_f32) or ("fma.rn.ftz.f32x2" in s8_f32)
        )
        assert_false(
            "fma.rn.f32 " in _compile_code[_mul_with_fastmath_none[width=1]]()
        )
    else:
        var s1_f16 = _compile_code[fma[width=1]]()
        var s2_f16 = _compile_code[fma[width=2]]()
        var s8_f16 = _compile_code[fma[width=8]]()
        # Allow FTZ only on Hopper+ based on the compiled accelerator arch.

        assert_true(("fma.rn.f16" in s1_f16) or ("fma.rn.ftz.f16" in s1_f16))
        assert_true(
            ("fma.rn.f16x2" in s2_f16) or ("fma.rn.ftz.f16x2" in s2_f16)
        )
        assert_true(
            ("fma.rn.f16x2" in s8_f16) or ("fma.rn.ftz.f16x2" in s8_f16)
        )
        assert_false(
            "fma.rn.f16 " in _compile_code[_mul_with_fastmath_none[width=1]]()
        )


def test_cast():
    fn cast[
        src_type: DType, dst_type: DType, width: Int
    ](src: SIMD[src_type, width]) -> SIMD[dst_type, width]:
        return src.cast[dst_type]()

    var s_f32_to_f16x2 = _compile_code[
        cast[src_type = DType.float32, dst_type = DType.float16, width=4]
    ]()
    # Allow optional .ftz in cvt.* on Hopper+ backends.
    assert_true(
        ("cvt.rn.f16x2.f32" in s_f32_to_f16x2)
        or ("cvt.rn.ftz.f16x2.f32" in s_f32_to_f16x2)
    )

    var s_f32_to_bf16x2 = _compile_code[
        cast[src_type = DType.float32, dst_type = DType.bfloat16, width=4]
    ]()
    # Allow optional .ftz in cvt.* on Hopper+ backends.
    assert_true(
        ("cvt.rn.bf16x2.f32" in s_f32_to_bf16x2)
        or ("cvt.rn.ftz.bf16x2.f32" in s_f32_to_bf16x2)
    )
    var s_b2f_1 = _compile_code[
        cast[src_type = DType.bfloat16, dst_type = DType.float32, width=1]
    ]()
    # Allow optional .ftz in cvt.* on Hopper+ backends.
    assert_true(("cvt.f32.bf16" in s_b2f_1) or ("cvt.ftz.f32.bf16" in s_b2f_1))

    var s_b2f_4 = _compile_code[
        cast[src_type = DType.bfloat16, dst_type = DType.float32, width=4]
    ]()
    # Allow optional .ftz in cvt.* on Hopper+ backends.
    assert_true(("cvt.f32.bf16" in s_b2f_4) or ("cvt.ftz.f32.bf16" in s_b2f_4))


def main():
    test_half_float_instruction_selection()

    test_fma[DType.bfloat16]()
    test_fma[DType.float16]()

    test_cast()

    alias device = GPUInfo.from_name[_accelerator_arch()]()

    @parameter
    if device == B200:
        test_add[DType.float32, "sm_100"]()
        test_mul[DType.float32, "sm_100"]()
        test_fma[DType.float32]()
