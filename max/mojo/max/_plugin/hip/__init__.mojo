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

from std.atomic import Ordering, fence
from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.sys import bit_width_of, llvm_intrinsic
from std.sys.defines import get_defined_bool
from std.sys.info import CompilationTarget

from max.gpu.intrinsics import readfirstlane
from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from .._shuffle import shuffle_words
from .._trait import MaxPlugin, MaxPluginHooks

comptime _USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM = get_defined_bool[
    "USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM", False
]()


@inline(.always)
def _barrier():
    comptime if _USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM:
        llvm_intrinsic["llvm.amdgcn.s.waitcnt", NoneType](Int32(0xC07F))
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
    else:
        fence[Ordering.RELEASE, scope="workgroup"]()
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
        fence[Ordering.ACQUIRE, scope="workgroup"]()


@inline(.always)
def _syncwarp(mask: Int):
    llvm_intrinsic["llvm.amdgcn.wave.barrier", NoneType]()


@inline(.always)
def _shuffle_word(mask: UInt, word: UInt32, offset: UInt32) -> UInt32:
    # FIXME: Set the EXECute mask register to the mask
    return llvm_intrinsic["llvm.amdgcn.ds.bpermute", UInt32](offset * 4, word)


struct HipMaxPlugin(MaxPlugin):
    """Provides default MAX behavior for HIP targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["hip"]()
    comptime HookFor[target: CompilationTarget] = HipMaxPluginHooks[target]


struct HipMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    # gfx9 (GCN and CDNA 2 through 4) runs wave64; gfx10 and later, RDNA and
    # CDNA 5, run wave32.
    comptime warp_size = 64 if Self.target._arch().startswith("gfx9") else 32
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp

    @inline(.always)
    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return shuffle_words[_shuffle_word](mask, val, offset)

    @inline(.always)
    @staticmethod
    def vote_fn[ret_type: DType](val: Bool) -> Scalar[ret_type]:
        comptime instruction = String(
            "llvm.amdgcn.ballot.i", bit_width_of[ret_type]()
        )
        return llvm_intrinsic[
            instruction,
            Scalar[ret_type],
            has_side_effect=False,
        ](val)

    @inline(.always)
    @staticmethod
    def match_any_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        # CDNA has no match op, so fold with ROCm's `__match_any` idiom
        # (amd_warp_sync_functions.h): loop while any lane is still unmatched;
        # each round the lowest unmatched lane broadcasts its bits via
        # `readfirstlane` and a ballot over the unmatched lanes picks out the
        # ones that match -- their group -- which then drop out.  (ROCm reads
        # the group off `__activemask()` under branch divergence; a converged
        # ballot of the match predicate is the same set and survives the Mojo ->
        # LLVM lowering.)  O(distinct values), far below `WARP_SIZE` for the few
        # distinct keys a warp usually holds.
        var done = False
        var result = Scalar[mask_type](0)
        while Self.vote_fn[mask_type](not done) != Scalar[mask_type](0):
            if not done:
                var matches = readfirstlane(bits) == bits
                var group = Self.vote_fn[mask_type](matches)
                if matches:
                    result = group
                    done = True
        return result

    @inline(.always)
    @staticmethod
    def match_all_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        # All lanes agree iff the lanes matching the lowest lane's bits
        # (`readfirstlane`) are exactly the active lanes.
        var chosen = readfirstlane(bits)
        var matched = Self.vote_fn[mask_type](bits == chosen)
        var active = Self.vote_fn[mask_type](True)
        return matched if matched == active else Scalar[mask_type](0)
