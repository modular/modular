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

from math import exp2

from gpu.host.compile import _compile_code
from gpu.host.info import A100
from gpu.intrinsics import *


fn kernel[
    dtype: DType, memory: Bool = True
](
    output: UnsafePointer[Scalar[dtype]],
    ptr: UnsafePointer[Scalar[dtype]],
    val: Scalar[dtype],
):
    store_release[memory=memory](ptr, val)
    output[] = load_acquire[memory=memory](ptr)


# CHECK-LABEL: test_compile_code
def test_compile_code():
    print("== test_compile_code")

    # CHECK: st.release.sys.global.u32 [%rd1], %r1;
    # CHECK: ld.acquire.sys.global.u32 %r2, [%rd1];
    print(_compile_code[kernel[DType.int32], target = A100.target()]())

    # CHECK: st.release.sys.global.u16 [%rd1], %rs1;
    # CHECK: ld.acquire.sys.global.u16 %rs2, [%rd1];
    print(_compile_code[kernel[DType.bfloat16], target = A100.target()]())

    # CHECK: st.release.sys.global.u32 [%rd1], %r1;
    # CHECK: ld.acquire.sys.global.u32 %r2, [%rd1];
    print(
        _compile_code[
            kernel[DType.int32, memory=False], target = A100.target()
        ]()
    )

    # CHECK: st.release.sys.global.u16 [%rd1], %rs1;
    # CHECK: ld.acquire.sys.global.u16 %rs2, [%rd1];
    print(
        _compile_code[
            kernel[DType.bfloat16, memory=False], target = A100.target()
        ]()
    )

    # CHECK: tail call void asm sideeffect "st.release.sys.global.u16 [$1], $0;", "h,l,~{memory}"(bfloat %2, ptr %1)
    # CHECK: tail call bfloat asm sideeffect "ld.acquire.sys.global.u16 $0, [$1];", "=h,l,~{memory}"(ptr %1)
    print(
        _compile_code[
            kernel[DType.bfloat16, memory=True],
            target = A100.target(),
            emission_kind="llvm-opt",
        ]()
    )

    # CHECK: tail call void asm sideeffect "st.release.sys.global.u16 [$1], $0;", "h,l"(bfloat %2, ptr %1)
    # CHECK: tail call bfloat asm sideeffect "ld.acquire.sys.global.u16 $0, [$1];", "=h,l"(ptr %1)
    print(
        _compile_code[
            kernel[DType.bfloat16, memory=False],
            target = A100.target(),
            emission_kind="llvm-opt",
        ]()
    )

    # https://godbolt.org/z/j9ecfjjP1
    fn exp_op(output: UnsafePointer[Float32], max_scaled: Int32):
        output[] = exp2(
            output[] * 1.44269504088896340736 - max_scaled.cast[DType.float32]()
        )

    # CHECK: "target-cpu"="sm_80" "target-features"="+ptx81,+sm_80" "tune-cpu"="sm_80"
    print(
        _compile_code[
            exp_op, target = A100.target(), emission_kind="llvm-opt"
        ]()
    )
    # CHECK: fma.rn.f32
    print(_compile_code[exp_op, target = A100.target(), emission_kind="asm"]())


def main():
    test_compile_code()
