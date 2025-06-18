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
from sys.info import CompilationTarget, simdwidthof
from sys.intrinsics import llvm_intrinsic


@always_inline
fn roundeven_to_int32[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[DType.int32, simd_width]:
    alias native_width = simdwidthof[type]()

    # Use the AVX512 instruction `vcvtps2dq` with embedded rounding control
    # set to do rounding to nearest with ties to even (roundeven). This
    # replaces a `vrndscaleps` and `vcvttps2dq` instruction pair.
    @parameter
    if (
        CompilationTarget.has_avx512f()
        and type is DType.float32
        and simd_width >= native_width
    ):
        var x_i32 = SIMD[DType.int32, simd_width]()

        @parameter
        for i in range(0, simd_width, native_width):
            var part = llvm_intrinsic[
                "llvm.x86.avx512.mask.cvtps2dq.512",
                SIMD[DType.int32, native_width],
                has_side_effect=False,
            ](
                x.slice[native_width, offset=i](),
                SIMD[DType.int32, native_width](0),
                Int16(-1),  # no mask
                Int32(8),  # round to nearest
            )
            x_i32 = x_i32.insert[offset=i](part)

        return x_i32

    # Use the NEON instruction `fcvtns` to fuse the conversion to int32
    # with rounding to nearest with ties to even (roundeven). This
    # replaces a `frintn` and `fcvtzs` instruction pair.
    @parameter
    if (
        CompilationTarget.has_neon()
        and type is DType.float32
        and simd_width >= native_width
    ):
        var x_i32 = SIMD[DType.int32, simd_width]()

        @parameter
        for i in range(0, simd_width, native_width):
            var part = llvm_intrinsic[
                "llvm.aarch64.neon.fcvtns.v4i32.v4f32",
                SIMD[DType.int32, native_width],
                has_side_effect=False,
            ](x.slice[native_width, offset=i]())
            x_i32 = x_i32.insert[offset=i](part)

        return x_i32

    return round(x).cast[DType.int32]()
