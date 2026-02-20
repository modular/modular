# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from sys import llvm_intrinsic

from memory.unsafe import bitcast

# ===-----------------------------------------------------------------------===#
# dot product
# ===-----------------------------------------------------------------------===#


fn _neon_dotprod[
    a_type: DType, b_type: DType, c_type: DType, width: Int
](
    c: SIMD[c_type, width],
    a: SIMD[a_type, width * 4],
    b: SIMD[b_type, width * 4],
) -> SIMD[c_type, width]:
    comptime assert c_type == DType.int32, "the type of C must be int32"
    comptime assert width == 4

    @parameter
    @always_inline
    fn call_intrinsic[intrin: StaticString]() -> SIMD[c_type, width]:
        return llvm_intrinsic[intrin, SIMD[c_type, width]](c, a, b)

    comptime if a_type == DType.uint8 and b_type == DType.uint8:
        return call_intrinsic["llvm.aarch64.neon.udot.v4i32.v16i8"]()
    elif a_type == DType.int8 and b_type == DType.int8:
        return call_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8"]()
    else:
        constrained[False, "unsupported A and B types"]()
        return SIMD[c_type, width]()


fn _neon_dotprod_lane[
    lane: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    width: Int,
    b_width: Int,
](
    c: SIMD[c_type, width],
    a: SIMD[a_type, width * 4],
    b: SIMD[b_type, b_width],
) -> SIMD[c_type, width]:
    comptime assert (
        b_type == DType.int8 or b_type == DType.uint8
    ), "unsupported B type"
    comptime assert 4 <= b_width <= 16, "unsupported B width"
    comptime assert 0 <= lane < (b_width // 4), "invalid lane index"

    # Helper to generate `sdot r, a, b[lane]` instruction form.
    var tuple = bitcast[DType.int32, b_width // 4](b)[lane]
    var splat = bitcast[b_type, width * 4](SIMD[DType.int32, width](tuple))
    return _neon_dotprod(c, a, splat)


# ===-----------------------------------------------------------------------===#
# matrix multiply-accumulate
# ===-----------------------------------------------------------------------===#


fn _neon_matmul[
    a_type: DType, b_type: DType, c_type: DType, width: Int
](
    c: SIMD[c_type, width],
    a: SIMD[a_type, width * 4],
    b: SIMD[b_type, width * 4],
) -> SIMD[c_type, width]:
    comptime assert c_type == DType.int32, "the type of C must be int32"
    comptime assert width == 4

    @parameter
    @always_inline
    fn call_intrinsic[intrin: StaticString]() -> SIMD[c_type, width]:
        return llvm_intrinsic[intrin, SIMD[c_type, width]](c, a, b)

    comptime if a_type == DType.uint8 and b_type == DType.uint8:
        return call_intrinsic["llvm.aarch64.neon.ummla.v4i32.v16i8"]()
    elif a_type == DType.uint8 and b_type == DType.int8:
        return call_intrinsic["llvm.aarch64.neon.usmmla.v4i32.v16i8"]()
    elif a_type == DType.int8 and b_type == DType.int8:
        return call_intrinsic["llvm.aarch64.neon.smmla.v4i32.v16i8"]()
    else:
        constrained[False, "unsupported A and B types"]()
        return SIMD[c_type, width]()
