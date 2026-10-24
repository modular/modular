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
from std.sys import llvm_intrinsic, size_of
from std.sys._assembly import inlined_assembly
from std.sys.info import CompilationTarget

from max.gpu.globals import WARP_SIZE
from max.gpu.memory.memory import (
    _AsyncCopyGroupPluginHookFnType,
    _AsyncCopyWaitGroupPluginHookFnType,
)
from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from .._shuffle import shuffle_words
from .._trait import MaxPlugin, MaxPluginHooks


@inline(.always)
def _barrier():
    _ = __mlir_op.`nvvm.barrier`[
        _properties=__mlir_attr.`{operandSegmentSizes = array<i32: 0, 0>}`,
    ]()


@inline(.always)
def _syncwarp(mask: Int):
    __mlir_op.`nvvm.bar.warp.sync`(
        __mlir_op.`index.casts`[_type=__mlir_type.i32](mask.__mlir_index__())
    )


@inline(.always)
def _async_copy_commit_group():
    llvm_intrinsic["llvm.nvvm.cp.async.commit.group", NoneType]()


@inline(.always)
def _async_copy_wait_group(n: Int32):
    llvm_intrinsic["llvm.nvvm.cp.async.wait.group", NoneType](n)


@inline(.always)
def _async_copy_wait_all():
    llvm_intrinsic["llvm.nvvm.cp.async.wait.all", NoneType]()


# TODO (#24457): support shuffles with width != 32
comptime _WIDTH_MASK = WARP_SIZE - 1

# shfl.sync.up.b32 prepares this mask differently from other shuffle intrinsics
comptime _WIDTH_MASK_SHUFFLE_UP = 0


@inline(.always)
def _shuffle_word[
    mnemonic: StaticString, width_mask: Int32
](mask: UInt, word: UInt32, offset: UInt32) -> UInt32:
    return llvm_intrinsic["llvm.nvvm.shfl.sync." + mnemonic + ".i32", UInt32](
        Int32(mask), word, offset, width_mask
    )


struct CudaMaxPlugin(MaxPlugin):
    """Provides default MAX behavior for CUDA targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["cuda"]()
    comptime HookFor[target: CompilationTarget] = CudaMaxPluginHooks[target]


struct CudaMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime warp_size = 32
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp
    comptime async_copy_commit_group_fn: OptionalReg[
        _AsyncCopyGroupPluginHookFnType
    ] = _async_copy_commit_group
    comptime async_copy_wait_group_fn: OptionalReg[
        _AsyncCopyWaitGroupPluginHookFnType
    ] = _async_copy_wait_group
    comptime async_copy_wait_all_fn: OptionalReg[
        _AsyncCopyGroupPluginHookFnType
    ] = _async_copy_wait_all

    @inline(.always)
    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return shuffle_words[_shuffle_word["idx", Int32(_WIDTH_MASK)]](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def shuffle_up_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return shuffle_words[_shuffle_word["up", _WIDTH_MASK_SHUFFLE_UP]](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def shuffle_down_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return shuffle_words[_shuffle_word["down", Int32(_WIDTH_MASK)]](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def shuffle_xor_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return shuffle_words[_shuffle_word["bfly", Int32(_WIDTH_MASK)]](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def vote_fn[ret_type: DType](val: Bool) -> Scalar[ret_type]:
        comptime assert ret_type == .uint32, "Unsupported return type"
        return rebind[Scalar[ret_type]](
            llvm_intrinsic[
                "llvm.nvvm.vote.ballot.sync",
                UInt32,
                UInt32,
                Bool,
                has_side_effect=False,
            ](0xFFFFFFFF, val).cast[.uint32]()
        )

    @inline(.always)
    @staticmethod
    def match_any_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        comptime assert (
            mask_type == .uint32
        ), "NVIDIA match_any returns a 32-bit mask (mask_type == DType.uint32)"
        comptime if size_of[dtype]() == 4:
            return rebind[Scalar[mask_type]](
                inlined_assembly[
                    "match.any.sync.b32 $0, $1, $2;",
                    UInt32,
                    constraints="=r,r,r",
                    has_side_effect=False,
                ](bits, UInt32(0xFFFFFFFF))
            )
        else:
            return rebind[Scalar[mask_type]](
                inlined_assembly[
                    "match.any.sync.b64 $0, $1, $2;",
                    UInt32,
                    constraints="=r,l,r",
                    has_side_effect=False,
                ](bits, UInt32(0xFFFFFFFF))
            )

    @inline(.always)
    @staticmethod
    def match_all_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        comptime assert (
            mask_type == .uint32
        ), "NVIDIA match_all returns a 32-bit mask (mask_type == DType.uint32)"
        # `match.all.sync` writes the membermask (all agree) or 0 into `$0` and
        # the agreement into a predicate `p` we do not need (the mask already
        # encodes it as non-zero / zero).
        comptime if size_of[dtype]() == 4:
            return rebind[Scalar[mask_type]](
                inlined_assembly[
                    "{ .reg .pred p; match.all.sync.b32 $0|p, $1, $2; }",
                    UInt32,
                    constraints="=r,r,r",
                    has_side_effect=False,
                ](bits, UInt32(0xFFFFFFFF))
            )
        else:
            return rebind[Scalar[mask_type]](
                inlined_assembly[
                    "{ .reg .pred p; match.all.sync.b64 $0|p, $1, $2; }",
                    UInt32,
                    constraints="=r,l,r",
                    has_side_effect=False,
                ](bits, UInt32(0xFFFFFFFF))
            )
