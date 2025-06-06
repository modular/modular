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

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from memory import UnsafePointer
from testing import assert_false, assert_true

alias A100_TARGET = _get_gpu_target["sm_80"]()
alias MI300X_TARGET = _get_gpu_target["mi300x"]()


def test_abs():
    fn do_abs[
        type: DType, *, width: Int = 1
    ](val: SIMD[type, width]) -> __type_of(val):
        return abs(val)

    # AMD GPU kernels cannot have a return value
    fn do_abs_noreturn[
        type: DType, *, width: Int = 1
    ](val: SIMD[type, width], x: UnsafePointer[Scalar[type]]):
        x.store(0, abs(val))

    # Check the NVIDIA PTX.
    assert_true(
        "abs.f16 "
        in _compile_code_asm[do_abs[DType.float16], target=A100_TARGET]()
    )
    assert_true(
        "abs.bf16 "
        in _compile_code_asm[do_abs[DType.bfloat16], target=A100_TARGET]()
    )

    assert_true(
        "abs.f16x2 "
        in _compile_code_asm[
            do_abs[DType.float16, width=2], target=A100_TARGET
        ]()
    )
    assert_true(
        "abs.bf16x2 "
        in _compile_code_asm[
            do_abs[DType.bfloat16, width=2], target=A100_TARGET
        ]()
    )

    # Check the AMD CDNA assembly.

    # Set the sign bit to zero.
    assert_true(
        "s_and_b32 s0, s4, 0x7fffffff"
        in _compile_code_asm[
            do_abs_noreturn[DType.float32], target=MI300X_TARGET
        ]()
    )

    # Mask out the lower half sign bit.
    assert_true(
        "s_and_b32 s0, s4, 0x7fff"
        in _compile_code_asm[
            do_abs_noreturn[DType.float16, width=1], target=MI300X_TARGET
        ]()
    )
    # Mask out the lower and upper half sign bit
    assert_true(
        "s_and_b32 s0, s4, 0x7fff7fff"
        in _compile_code_asm[
            do_abs_noreturn[DType.float16, width=2], target=MI300X_TARGET
        ]()
    )
    # Mask out the sign bit.
    assert_true(
        "s_and_b32 s0, s4, 0x7fff"
        in _compile_code_asm[
            do_abs_noreturn[DType.bfloat16, width=1], target=MI300X_TARGET
        ]()
    )
    # Should be able to mask with 0x7fff7fff
    # but instead it does low and high halves sequentially. Like this
    # s_and_b32 s3, s2, 0x7fff
    # s_lshr_b32 s2, s2, 16
    # s_and_b32 s2, s2, 0x7fff
    assert_false(
        "s_and_b32 s0, s4, 0x7fff7fff"
        in _compile_code_asm[
            do_abs_noreturn[DType.bfloat16, width=2], target=MI300X_TARGET
        ]()
    )


def main():
    test_abs()
