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

from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.memory import bitcast
from std.sys import bit_width_of, llvm_intrinsic
from std.sys.info import CompilationTarget

from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from .._trait import MaxPlugin, MaxPluginHooks


@inline(.always)
def _barrier():
    # threadgroup_barrier(mem_flags::mem_threadgroup)
    llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(2), Int32(1))


@inline(.always)
def _syncwarp(mask: Int):
    # simdgroup_barrier(mem_flags::mem_none)
    llvm_intrinsic["llvm.air.simdgroup.barrier", NoneType](Int32(0), Int32(4))


@inline(.always)
def _shuffle_apple_helper[
    op: StringSlice, dtype: DType, simd_width: SIMDLength
](
    mask: UInt,  # Ignored, for API parity
    val: SIMD[dtype, simd_width],
    offset: UInt32,
) -> SIMD[dtype, simd_width]:
    """
    Mapping from Metal stdlib to AIR (LLVM) intrinsics:
      Metal                         → AIR intrinsic stem
      ----------------------------------------------------------
      simd_shuffle                  → llvm.air.simd_shuffle
      simd_shuffle_down             → llvm.air.simd_shuffle_down
      simd_shuffle_up               → llvm.air.simd_shuffle_up
      simd_shuffle_xor              → llvm.air.simd_shuffle_xor
    """

    comptime assert (
        dtype.is_half_float() or simd_width == 1
    ), "Unsupported simd_width"

    var arg = UInt16(offset)  # AIR intrinsics use 16-bit offsets

    comptime if dtype.is_integral() and bit_width_of[dtype]() == 64:
        var bits = bitcast[.uint32, simd_width * 2](val)
        var half1, half2 = bits.deinterleave()

        var half1_n = rebind[SIMD[.uint32, simd_width]](half1)
        var half2_n = rebind[SIMD[.uint32, simd_width]](half2)
        var s1 = _shuffle_apple_helper[op, DType.uint32, simd_width](
            mask, half1_n, offset
        )
        var s2 = _shuffle_apple_helper[op, DType.uint32, simd_width](
            mask, half2_n, offset
        )

        var merged = s1.interleave(s2)
        return bitcast[dtype, simd_width](merged)
    elif dtype == .bool:
        var val1 = rebind[Int32](val.cast[.int32]())
        var tmp = _shuffle_apple_helper[op, DType.int32, 1](mask, val1, offset)
        return tmp.cast[dtype]()
    elif (
        dtype == .bfloat16
    ):  # bfloat16 is declared in MSL but actually causes a backend error.
        comptime if simd_width == 1:
            var pair = SIMD[dtype, 2](val._refine[new_size=1]())
            var pair_i32 = bitcast[.int32, 1](pair)
            var y_i32 = _shuffle_apple_helper[op, DType.int32, 1](
                mask, pair_i32, offset
            )
            return bitcast[dtype, 2](y_i32)[0]
        else:
            var packed = bitcast[.int32, simd_width // 2](val)
            var packed_shuf = _shuffle_apple_helper[
                op, DType.int32, simd_width // 2
            ](mask, packed, offset)
            return bitcast[dtype, simd_width](packed_shuf)
    else:
        comptime name = "llvm.air.simd_shuffle" + (
            "" if op == "indexed" else "_" + op
        )
        return llvm_intrinsic[name, SIMD[dtype, simd_width]](val, arg)


struct MetalMaxPlugin(MaxPlugin):
    """Provides default MAX behavior for Metal targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()
    comptime HookFor[target: CompilationTarget] = MetalMaxPluginHooks[target]


struct MetalMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime warp_size = 32
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp

    @inline(.always)
    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return _shuffle_apple_helper["indexed", dtype, simd_width](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def shuffle_up_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return _shuffle_apple_helper["up", dtype, simd_width](mask, val, offset)

    @inline(.always)
    @staticmethod
    def shuffle_down_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return _shuffle_apple_helper["down", dtype, simd_width](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def shuffle_xor_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return _shuffle_apple_helper["xor", dtype, simd_width](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def vote_fn[ret_type: DType](val: Bool) -> Scalar[ret_type]:
        # AIR has a dedicated 32-bit ballot intrinsic; narrowing the 64-bit
        # `simd_ballot`/`simd_vote` form instead crashes the Metal shader
        # compiler at pipeline-state creation.
        var mask32 = llvm_intrinsic["llvm.air.simd_ballot.i32", UInt32](val)
        return mask32.cast[ret_type]()
